# Copyright Anne M. Archibald 2008
# Additional contributions by Patrick Varilly and Sturla Molden 2012
# Revision by Sturla Molden 2015
# Balanced kd-tree construction written by Jake Vanderplas for scikit-learn
# Released under the scipy license

# cython: cpow=True


import mlx.core as mx
import scipy.sparse
from scipy_mlx._lib._util import copy_if_needed

cimport mlx.core as mx

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport isinf, INFINITY

cimport cython
import os
import threading
import operator

mx.import_array()

cdef extern from "<limits.h>":
    long LONG_MAX


__all__ = ['cKDTree']


# C++ implementations
# ===================

cdef extern from "ckdtree_decl.h":
    struct ckdtreenode:
        mx.intp_t split_dim
        mx.intp_t children
        mx.float64_t split
        mx.intp_t start_idx
        mx.intp_t end_idx
        ckdtreenode *less
        ckdtreenode *greater
        mx.intp_t _less
        mx.intp_t _greater

    struct ckdtree:
        vector[ckdtreenode]  *tree_buffer
        ckdtreenode   *ctree
        mx.float64_t   *raw_data
        mx.intp_t      n
        mx.intp_t      m
        mx.intp_t      leafsize
        mx.float64_t   *raw_maxes
        mx.float64_t   *raw_mins
        mx.intp_t      *raw_indices
        mx.float64_t   *raw_boxsize_data
        mx.intp_t size

    # External build and query methods in C++.

    int build_ckdtree(ckdtree *self,
                         mx.intp_t start_idx,
                         mx.intp_t end_idx,
                         mx.float64_t *maxes,
                         mx.float64_t *mins,
                         int _median,
                         int _compact) except + nogil

    int build_weights(ckdtree *self,
                         mx.float64_t *node_weights,
                         mx.float64_t *weights) except + nogil

    int query_knn(const ckdtree *self,
                     mx.float64_t *dd,
                     mx.intp_t    *ii,
                     const mx.float64_t *xx,
                     const mx.intp_t    n,
                     const mx.intp_t    *k,
                     const mx.intp_t    nk,
                     const mx.intp_t    kmax,
                     const mx.float64_t eps,
                     const mx.float64_t p,
                     const mx.float64_t distance_upper_bound) except + nogil

    int query_pairs(const ckdtree *self,
                       const mx.float64_t r,
                       const mx.float64_t p,
                       const mx.float64_t eps,
                       vector[ordered_pair] *results) except + nogil

    int count_neighbors_unweighted(const ckdtree *self,
                           const ckdtree *other,
                           mx.intp_t     n_queries,
                           mx.float64_t  *real_r,
                           mx.intp_t     *results,
                           const mx.float64_t p,
                           int cumulative) except + nogil

    int count_neighbors_weighted(const ckdtree *self,
                           const ckdtree *other,
                           mx.float64_t  *self_weights,
                           mx.float64_t  *other_weights,
                           mx.float64_t  *self_node_weights,
                           mx.float64_t  *other_node_weights,
                           mx.intp_t     n_queries,
                           mx.float64_t  *real_r,
                           mx.float64_t     *results,
                           const mx.float64_t p,
                           int cumulative) except + nogil

    int query_ball_point(const ckdtree *self,
                         const mx.float64_t *x,
                         const mx.float64_t *r,
                         const mx.float64_t p,
                         const mx.float64_t eps,
                         const mx.intp_t n_queries,
                         vector[mx.intp_t] *results,
                         const bool return_length,
                         const bool sort_output) except + nogil

    int query_ball_tree(const ckdtree *self,
                           const ckdtree *other,
                           const mx.float64_t r,
                           const mx.float64_t p,
                           const mx.float64_t eps,
                           vector[mx.intp_t] *results) except + nogil

    int sparse_distance_matrix(const ckdtree *self,
                                  const ckdtree *other,
                                  const mx.float64_t p,
                                  const mx.float64_t max_distance,
                                  vector[coo_entry] *results) except + nogil


# C++ helper functions
# ====================

cdef extern from "coo_entries.h":

    struct coo_entry:
        mx.intp_t i
        mx.intp_t j
        mx.float64_t v

cdef extern from "ordered_pair.h":

    struct ordered_pair:
        mx.intp_t i
        mx.intp_t j

# coo_entry wrapper
# =================

cdef class coo_entries:

    cdef:
        readonly object __array_interface__
        vector[coo_entry] *buf

    def __cinit__(coo_entries self):
        self.buf = NULL

    def __init__(coo_entries self):
        self.buf = new vector[coo_entry]()

    def __dealloc__(coo_entries self):
        if self.buf != NULL:
            del self.buf

    # The methods array, dict, coo_matrix, and dok_matrix must only
    # be called after the buffer is filled with coo_entry data. This
    # is because std::vector can reallocate its internal buffer when
    # push_back is called.

    def array(coo_entries self):
        cdef:
            coo_entry *pr
            mx.uintp_t uintptr
            mx.intp_t n
        _dtype = [('i',mx.intp),('j',mx.intp),('v',mx.float64)]
        res_dtype = mx.dtype(_dtype, align = True)
        n = <mx.intp_t> self.buf.size()
        if (n > 0):
            pr = self.buf.data()
            uintptr = <mx.uintp_t> (<void*> pr)
            dtype = mx.dtype(mx.uint8)
            self.__array_interface__ = dict(
                data = (uintptr, False),
                descr = dtype.descr,
                shape = (n*sizeof(coo_entry),),
                strides = (dtype.itemsize,),
                typestr = dtype.str,
                version = 3,
            )
            return mx.array(self).view(dtype=res_dtype)
        else:
            return mx.empty(shape=(0,), dtype=res_dtype)

    def dict(coo_entries self):
        cdef:
            mx.intp_t i, j, k, n
            mx.float64_t v
            coo_entry *pr
            dict res_dict
        n = <mx.intp_t> self.buf.size()
        if (n > 0):
            pr = self.buf.data()
            res_dict = dict()
            for k in range(n):
                i = pr[k].i
                j = pr[k].j
                v = pr[k].v
                res_dict[(i,j)] = v
            return res_dict
        else:
            return {}

    def coo_matrix(coo_entries self, m, n):
        res_arr = self.array()
        return scipy.sparse.coo_matrix(
                       (res_arr['v'], (res_arr['i'], res_arr['j'])),
                                       shape=(m, n))

    def dok_matrix(coo_entries self, m, n):
        return self.coo_matrix(m,n).todok()


# ordered_pair wrapper
# ====================

cdef class ordered_pairs:

    cdef:
        readonly object __array_interface__
        vector[ordered_pair] *buf

    def __cinit__(ordered_pairs self):
        self.buf = NULL

    def __init__(ordered_pairs self):
        self.buf = new vector[ordered_pair]()

    def __dealloc__(ordered_pairs self):
        if self.buf != NULL:
            del self.buf

    # The methods array and set must only be called after the buffer
    # is filled with ordered_pair data.

    def array(ordered_pairs self):
        cdef:
            ordered_pair *pr
            mx.uintp_t uintptr
            mx.intp_t n
        n = <mx.intp_t> self.buf.size()
        if (n > 0):
            pr = self.buf.data()
            uintptr = <mx.uintp_t> (<void*> pr)
            dtype = mx.dtype(mx.intp)
            self.__array_interface__ = dict(
                data = (uintptr, False),
                descr = dtype.descr,
                shape = (n,2),
                strides = (2*dtype.itemsize,dtype.itemsize),
                typestr = dtype.str,
                version = 3,
            )
            return mx.array(self)
        else:
            return mx.empty(shape=(0,2), dtype=mx.intp)

    def set(ordered_pairs self):
        cdef:
            ordered_pair *pair
            mx.intp_t i, n
            set results
        results = set()
        pair = self.buf.data()
        n = <mx.intp_t> self.buf.size()
        # other platforms
        for i in range(n):
            results.add((pair.i, pair.j))
            pair += 1
        return results



# Tree structure exposed to Python
# ================================

cdef class cKDTreeNode:
    """
    class cKDTreeNode

    This class exposes a Python view of a node in the cKDTree object.

    All attributes are read-only.

    Attributes
    ----------
    level : int
        The depth of the node. 0 is the level of the root node.
    split_dim : int
        The dimension along which this node is split. If this value is -1
        the node is a leafnode in the kd-tree. Leafnodes are not split further
        and scanned by brute force.
    split : float
        The value used to separate split this node. Points with value >= split
        in the split_dim dimension are sorted to the 'greater' subnode
        whereas those with value < split are sorted to the 'lesser' subnode.
    children : int
        The number of data points sorted to this node.
    data_points : array of float64
        An array with the data points sorted to this node.
    indices : array of intp
        An array with the indices of the data points sorted to this node. The
        indices refer to the position in the data set used to construct the
        kd-tree.
    lesser : cKDTreeNode or None
        Subnode with the 'lesser' data points. This attribute is None for
        leafnodes.
    greater : cKDTreeNode or None
        Subnode with the 'greater' data points. This attribute is None for
        leafnodes.

    """
    cdef:
        readonly mx.intp_t    level
        readonly mx.intp_t    split_dim
        readonly mx.intp_t    children
        readonly mx.intp_t    start_idx
        readonly mx.intp_t    end_idx
        readonly mx.float64_t split
        mx.array            _data
        mx.array            _indices
        readonly object       lesser
        readonly object       greater

    cdef void _setup(cKDTreeNode self, cKDTree parent, ckdtreenode *node, mx.intp_t level) noexcept:
        cdef cKDTreeNode n1, n2
        self.level = level
        self.split_dim = node.split_dim
        self.children = node.children
        self.split = node.split
        self.start_idx = node.start_idx
        self.end_idx = node.end_idx
        self._data = parent.data
        self._indices = parent.indices
        if self.split_dim == -1:
            self.lesser = None
            self.greater = None
        else:
            # setup lesser branch
            n1 = cKDTreeNode()
            n1._setup(parent, node=node.less, level=level + 1)
            self.lesser = n1
            # setup greater branch
            n2 = cKDTreeNode()
            n2._setup(parent, node=node.greater, level=level + 1)
            self.greater = n2

    property data_points:
        def __get__(cKDTreeNode self):
            return self._data[self.indices,:]

    property indices:
        def __get__(cKDTreeNode self):
            cdef mx.intp_t start, stop
            start = self.start_idx
            stop = self.end_idx
            return self._indices[start:stop]


cdef mx.intp_t get_num_workers(workers: object, kwargs: dict) except -1:
    """Handle the workers argument"""
    if workers is None:
        workers = 1

    if len(kwargs) > 0:
        raise TypeError(
            f"Unexpected keyword argument{'s' if len(kwargs) > 1 else ''} "
            f"{kwargs}")

    cdef mx.intp_t n = operator.index(workers)
    if n == -1:
        num = os.cpu_count()
        if num is None:
            raise NotImplementedError(
                'Cannot determine the number of cpus using os.cpu_count(), '
                'cannot use -1 for the number of workers')
        n = num
    elif n <= 0:
        raise ValueError(f'Invalid number of workers {workers}, must be -1 or > 0')
    return n


# Main cKDTree class
# ==================

cdef class cKDTree:
    """
    cKDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
            balanced_tree=True, boxsize=None)

    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    .. note::
       `cKDTree` is functionally identical to `KDTree`. Prior to SciPy
       v1.6.0, `cKDTree` had better performance and slightly different
       functionality but now the two names exist only for
       backward-compatibility reasons. If compatibility with SciPy < 1.6 is not
       a concern, prefer `KDTree`.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force. Default: 16.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : array, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with ``copy_data=True``.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : array, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : array, shape (m,)
        The minimum value in each dimension of the n data points.
    tree : object, class cKDTreeNode
        This attribute exposes a Python view of the root node in the cKDTree
        object. A full Python view of the kd-tree is created dynamically
        on the first access. This attribute allows you to create your own
        query functions in Python.
    size : int
        The number of nodes in the tree.

    """
    cdef:
        ckdtree * cself
        object                   _python_tree
        readonly mx.array      data
        readonly mx.array      maxes
        readonly mx.array      mins
        readonly mx.array      indices
        readonly object          boxsize
        mx.array               boxsize_data

    property n:
        def __get__(self): return self.cself.n

    property m:
        def __get__(self): return self.cself.m

    property leafsize:
        def __get__(self): return self.cself.leafsize

    property size:
        def __get__(self): return self.cself.size

    property tree:
        # make the tree viewable from Python
        def __get__(cKDTree self):
            cdef cKDTreeNode n
            cdef ckdtree *cself = self.cself
            if self._python_tree is not None:
                return self._python_tree
            else:
                n = cKDTreeNode()
                n._setup(self, node=cself.ctree, level=0)
                self._python_tree = n
                return self._python_tree

    def __cinit__(cKDTree self):
        self.cself = <ckdtree * > PyMem_Malloc(sizeof(ckdtree))
        self.cself.tree_buffer = NULL

    def __init__(cKDTree self, data, mx.intp_t leafsize=16, compact_nodes=True,
            copy_data=False, balanced_tree=True, boxsize=None):

        cdef:
            mx.float64_t [::1] tmpmaxes, tmpmins
            mx.float64_t *ptmpmaxes
            mx.float64_t *ptmpmins
            ckdtree *cself = self.cself
            int compact, median

        self._python_tree = None

        if not copy_data:
            copy_data = copy_if_needed
        data = mx.array(data, order='C', copy=copy_data, dtype=mx.float64)

        if data.ndim != 2:
            raise ValueError("data must be of shape (n, m), where there are "
                             "n points of dimension m")

        if not mx.isfinite(data).all():
            raise ValueError("data must be finite, check for nan or inf values")

        self.data = data
        cself.n = data.shape[0]
        cself.m = data.shape[1]
        cself.leafsize = leafsize

        if leafsize<1:
            raise ValueError("leafsize must be at least 1")

        if boxsize is None:
            self.boxsize = None
            self.boxsize_data = None
        else:
            self.boxsize_data = mx.empty(2 * self.m, dtype=mx.float64)
            boxsize = broadcast_contiguous(boxsize, shape=(self.m,),
                                           dtype=mx.float64)
            self.boxsize_data[:self.m] = boxsize
            self.boxsize_data[self.m:] = 0.5 * boxsize

            self.boxsize = boxsize
            periodic_mask = self.boxsize > 0
            if ((self.data >= self.boxsize[None, :])[:, periodic_mask]).any():
                raise ValueError("Some input data are greater than the size of the periodic box.")
            if ((self.data < 0)[:, periodic_mask]).any():
                raise ValueError("Negative input data are outside of the periodic box.")

        self.maxes = mx.ascontiguousarray(
            mx.amax(self.data, axis=0) if self.n > 0 else mx.zeros(self.m),
            dtype=mx.float64)
        self.mins = mx.ascontiguousarray(
            mx.amin(self.data,axis=0) if self.n > 0 else mx.zeros(self.m),
            dtype=mx.float64)
        self.indices = mx.ascontiguousarray(mx.arange(self.n,dtype=mx.intp))

        self._pre_init()

        compact = 1 if compact_nodes else 0
        median = 1 if balanced_tree else 0

        cself.tree_buffer = new vector[ckdtreenode]()

        tmpmaxes = mx.copy(self.maxes)
        tmpmins = mx.copy(self.mins)

        ptmpmaxes = &tmpmaxes[0]
        ptmpmins = &tmpmins[0]
        with nogil:
            build_ckdtree(cself, 0, cself.n, ptmpmaxes, ptmpmins, median, compact)

        # set up the tree structure pointers
        self._post_init()

    cdef _pre_init(cKDTree self):
        cself = self.cself

        # finalize the pointers from array attributes

        cself.raw_data = <mx.float64_t*> mx.PyArray_DATA(self.data)
        cself.raw_maxes = <mx.float64_t*> mx.PyArray_DATA(self.maxes)
        cself.raw_mins = <mx.float64_t*> mx.PyArray_DATA(self.mins)
        cself.raw_indices = <mx.intp_t*> mx.PyArray_DATA(self.indices)

        if self.boxsize_data is not None:
            cself.raw_boxsize_data = <mx.float64_t*>mx.PyArray_DATA(self.boxsize_data)
        else:
            cself.raw_boxsize_data = NULL

    cdef _post_init(cKDTree self):
        cself = self.cself
        # finalize the tree points, this calls _post_init_traverse

        cself.ctree = cself.tree_buffer.data()

        # set the size attribute after tree_buffer is built
        cself.size = cself.tree_buffer.size()

        self._post_init_traverse(cself.ctree)

    cdef _post_init_traverse(cKDTree self, ckdtreenode *node):
        cself = self.cself
        # recurse the tree and re-initialize
        # "less" and "greater" fields
        if node.split_dim == -1:
            # leafnode
            node.less = NULL
            node.greater = NULL
        else:
            node.less = cself.ctree + node._less
            node.greater = cself.ctree + node._greater
            self._post_init_traverse(node.less)
            self._post_init_traverse(node.greater)

    def __dealloc__(cKDTree self):
        cself = self.cself
        if cself.tree_buffer != NULL:
            del cself.tree_buffer
        PyMem_Free(cself)

    # -----
    # query
    # -----

    @cython.boundscheck(False)
    def query(cKDTree self, object x, object k=1, mx.float64_t eps=0.0,
              mx.float64_t p=2.0, mx.float64_t distance_upper_bound=INFINITY,
              object workers=None, **kwargs):
        r"""
        query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=mx.inf, workers=1)

        Query the kd-tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : list of integer or integer
            The list of k-th nearest neighbors to return. If k is an
            integer it is treated as a list of [1, ... k] (range(1, k+1)).
            Note that the counting starts from 1.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
            A finite large p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is given
            all CPU threads are used. Default: 1.

            .. versionchanged:: 1.9.0
               The "n_jobs" argument was renamed "workers". The old name
               "n_jobs" was deprecated in SciPy 1.6.0 and was removed in
               SciPy 1.9.0.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape ``tuple+(k,)``.
            When k == 1, the last dimension of the output is squeezed.
            Missing neighbors are indicated with infinite distances.
        i : array of ints
            The index of each neighbor in ``self.data``.
            If ``x`` has shape ``tuple+(self.m,)``, then ``i`` has shape ``tuple+(k,)``.
            When k == 1, the last dimension of the output is squeezed.
            Missing neighbors are indicated with ``self.n``.

        Notes
        -----
        If the KD-Tree is periodic, the position ``x`` is wrapped into the
        box.

        When the input k is a list, a query for arange(max(k)) is performed, but
        only columns that store the requested values of k are preserved. This is
        implemented in a manner that reduces memory usage.

        Examples
        --------

        >>> import mlx.core as mx
        >>> from scipy_mlx.spatial import cKDTree
        >>> x, y = mx.mgrid[0:5, 2:8]
        >>> tree = cKDTree(mx.c_[x.ravel(), y.ravel()])

        To query the nearest neighbours and return squeezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
        >>> print(dd, ii, sep='\n')
        [2.         0.2236068]
        [ 0 13]

        To query the nearest neighbours and return unsqueezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
        >>> print(dd, ii, sep='\n')
        [[2.        ]
         [0.2236068]]
        [[ 0]
         [13]]

        To query the second nearest neighbours and return unsqueezed result,
        use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
        >>> print(dd, ii, sep='\n')
        [[2.23606798]
         [0.80622577]]
        [[ 6]
         [19]]

        To query the first and second nearest neighbours, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        or, be more specific

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        """

        cdef:
            mx.intp_t n
            const mx.float64_t [:, ::1] xx
            mx.array x_arr = mx.ascontiguousarray(x, dtype=mx.float64)
            ckdtree *cself = self.cself
            mx.intp_t num_workers = get_num_workers(workers, kwargs)

        n = num_points(x_arr, cself.m)
        xx = x_arr.reshape(n, cself.m)

        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")

        if not mx.isfinite(x_arr).all():
            raise ValueError("'x' must be finite, check for nan or inf values")

        cdef:
            bool single = (x_arr.ndim == 1)
            bool nearest = False

        if mx.isscalar(k):
            if k == 1:
                nearest = True
            k = mx.arange(1, k + 1)

        retshape = mx.shape(x_arr)[:-1]

        # The C++ function touches all dd and ii entries,
        # setting the missing values.

        cdef:
            mx.float64_t [:, ::1] dd = mx.empty((n,len(k)),dtype=mx.float64)
            mx.intp_t [:, ::1] ii = mx.empty((n,len(k)),dtype=mx.intp)
            mx.intp_t [::1] kk = mx.array(k, dtype=mx.intp)
            mx.intp_t kmax = mx.max(k)

        # Do the query in an external C++ function.
        def _thread_func(mx.intp_t start, mx.intp_t stop):
            cdef:
                mx.float64_t *pdd = &dd[start,0]
                mx.intp_t *pii = &ii[start,0]
                const mx.float64_t *pxx = &xx[start,0]
                mx.intp_t *pkk = &kk[0]
            with nogil:
                query_knn(cself, pdd, pii,
                    pxx, stop-start, pkk, kk.shape[0], kmax, eps, p, distance_upper_bound)

        _run_threads(_thread_func, n, num_workers)

        ddret = mx.reshape(dd, retshape + (len(k),))
        iiret = mx.reshape(ii, retshape + (len(k),))

        if nearest:
            ddret = ddret[..., 0]
            iiret = iiret[..., 0]
            # the only case where we return a python scalar
            if single:
                ddret = float(ddret)
                iiret = int(iiret)

        return ddret, iiret

    # ----------------
    # query_ball_point
    # ----------------

    def query_ball_point(cKDTree self, object x, object r,
                         mx.float64_t p=2.0, mx.float64_t eps=0.0,
                         object workers=None,
                         return_sorted=None,
                         return_length=False, **kwargs):
        """
        query_ball_point(self, x, r, p=2.0, eps=0.0, workers=1, return_sorted=None,
                         return_length=False)

        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : array_like, float
            The radius of points to return, shall broadcast to the length of x.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        workers : int, optional
            Number of jobs to schedule for parallel processing. If -1 is given
            all processors are used. Default: 1.

            .. versionchanged:: 1.9.0
               The "n_jobs" argument was renamed "workers". The old name
               "n_jobs" was deprecated in SciPy 1.6.0 and was removed in
               SciPy 1.9.0.

        return_sorted : bool, optional
            Sorts returned indices if True and does not sort them if False. If
            None, does not sort single point queries, but does sort
            multi-point queries which was the behavior before this option
            was added.

            .. versionadded:: 1.2.0
        return_length: bool, optional
            Return the number of points inside the radius instead of a list
            of the indices.
            .. versionadded:: 1.3.0

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a cKDTree and using
        query_ball_tree.

        Examples
        --------
        >>> import mlx.core as mx
        >>> from scipy import spatial
        >>> x, y = mx.mgrid[0:4, 0:4]
        >>> points = mx.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.cKDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]

        Query multiple points and plot the results:

        >>> import matplotlib.pyplot as plt
        >>> points = mx.array(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()

        """

        cdef:
            object[::1] vout
            mx.intp_t[::1] vlen
            mx.array x_arr = mx.ascontiguousarray(x, dtype=mx.float64)
            ckdtree *cself = self.cself
            bool rlen = return_length
            # compatibility with the old bug not sorting scalar queries.
            bool sort_output = return_sorted or (
                return_sorted is None and x_arr.ndim > 1)

            mx.intp_t num_workers = get_num_workers(workers, kwargs)
            mx.intp_t n = num_points(x_arr, cself.m)
            tuple retshape = mx.shape(x_arr)[:-1]
            mx.array r_arr = broadcast_contiguous(r, shape=retshape,
                                                    dtype=mx.float64)

            const mx.float64_t *vxx = <mx.float64_t*>x_arr.data
            const mx.float64_t *vrr = <mx.float64_t*>r_arr.data

        if not mx.isfinite(x_arr).all():
            raise ValueError("'x' must be finite, check for nan or inf values")

        if rlen:
            result = mx.empty(retshape, dtype=mx.intp)
            vlen = result.reshape(-1)
        else:
            result = mx.empty(retshape, dtype=object)
            vout = result.reshape(-1)

        def _thread_func(mx.intp_t start, mx.intp_t stop):
            cdef:
                vector[vector[mx.intp_t]] vvres
                mx.intp_t i, j, m
                mx.intp_t *cur
                const mx.float64_t *pvxx
                const mx.float64_t *pvrr
                list tmp

            vvres.resize(stop - start)
            pvxx = vxx + start * cself.m
            pvrr = vrr + start

            with nogil:
                query_ball_point(cself, pvxx,
                                 pvrr, p, eps, stop - start, vvres.data(),
                                 rlen, sort_output)

            for i in range(stop - start):
                if rlen:
                    vlen[start + i] = vvres[i].front()
                    continue

                m = <mx.intp_t> (vvres[i].size())
                tmp = m * [None]

                cur = vvres[i].data()
                for j in range(m):
                    tmp[j] = cur[j]
                vout[start + i] = tmp

        _run_threads(_thread_func, n, num_workers)

        if x_arr.ndim == 1: # scalar query, unpack result.
            result = result[()]
        return result

    # ---------------
    # query_ball_tree
    # ---------------

    def query_ball_tree(cKDTree self, cKDTree other,
                        mx.float64_t r, mx.float64_t p=2.0, mx.float64_t eps=0.0):
        """
        query_ball_tree(self, other, r, p=2.0, eps=0.0)

        Find all pairs of points between `self` and `other` whose distance is at most r

        Parameters
        ----------
        other : cKDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import mlx.core as mx
        >>> from scipy_mlx.spatial import cKDTree
        >>> rng = mx.random.default_rng()
        >>> points1 = rng.random((15, 2))
        >>> points2 = rng.random((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = cKDTree(points1)
        >>> kd_tree2 = cKDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...             [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """

        cdef:
            vector[vector[mx.intp_t]] vvres
            mx.intp_t i, j, n, m
            mx.intp_t *cur
            list results
            list tmp

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to query_ball_tree have different "
                             "dimensionality")

        n = self.n

        # allocate an array of std::vector<npy_intp>
        vvres.resize(n)

        # query in C++
        with nogil:
            query_ball_tree(self.cself, other.cself, r, p, eps, vvres.data())

        # store the results in a list of lists
        results = n * [None]
        for i in range(n):
            m = <mx.intp_t> (vvres[i].size())
            if (m > 0):
                tmp = m * [None]
                cur = vvres[i].data()
                for j in range(m):
                    tmp[j] = cur[j]
                results[i] = tmp
            else:
                results[i] = []

        return results

    # -----------
    # query_pairs
    # -----------

    def query_pairs(cKDTree self, mx.float64_t r, mx.float64_t p=2.0,
                    mx.float64_t eps=0.0, output_type='set'):
        """
        query_pairs(self, r, p=2.0, eps=0.0, output_type='set')

        Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  ``p`` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'array'. Default: 'set'

        Returns
        -------
        results : set or array
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close. If output_type is 'array', an array is
            returned instead of a set.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import mlx.core as mx
        >>> from scipy_mlx.spatial import cKDTree
        >>> rng = mx.random.default_rng()
        >>> points = rng.random((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = cKDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """

        cdef ordered_pairs results

        results = ordered_pairs()

        with nogil:
            query_pairs(self.cself, r, p, eps, results.buf)

        if output_type == 'set':
            return results.set()
        elif output_type == 'array':
            return results.array()
        else:
            raise ValueError("Invalid output type")

    def _build_weights(cKDTree self, object weights):
        """
        _build_weights(weights)

        Compute weights of nodes from weights of data points. This will sum
        up the total weight per node. This function is used internally.

        Parameters
        ----------
        weights : array_like
            weights of data points; must be the same length as the data points.
            currently only scalar weights are supported. Therefore the weights
            array must be 1 dimensional.

        Returns
        -------
        node_weights : array_like
            total weight for each KD-Tree node.

        """
        cdef:
            mx.intp_t num_of_nodes
            mx.float64_t [::1] node_weights
            mx.float64_t [::1] proper_weights
            mx.float64_t *pnw
            mx.float64_t *ppw

        num_of_nodes = self.cself.tree_buffer.size();
        node_weights = mx.empty(num_of_nodes, dtype=mx.float64)

        # FIXME: use templates to avoid the type conversion
        proper_weights = mx.ascontiguousarray(weights, dtype=mx.float64)

        if len(proper_weights) != self.n:
            raise ValueError('Number of weights differ from the number of data points')

        pnw = &node_weights[0]
        ppw = &proper_weights[0]

        with nogil:
            build_weights(self.cself, pnw, ppw)

        return node_weights

    # ---------------
    # count_neighbors
    # ---------------

    @cython.boundscheck(False)
    def count_neighbors(cKDTree self, cKDTree other, object r, mx.float64_t p=2.0,
                        object weights=None, int cumulative=True):
        """
        count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True)

        Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : cKDTree instance
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative (``cumulative=False``), ``r``
            defines the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in ``self``, and
            weights[1] is the weights of points in ``other``; either can be None to
            indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in ``self``
            and ``other``. For this to make sense, ``self`` and ``other`` must be the
            same tree. If ``self`` and ``other`` are two different trees, a ``ValueError``
            is raised.
            Default: None
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set to ``False``
            the algorithm is optimized to work with a large number of bins (>10) specified
            by ``r``. When ``cumulative`` is set to True, the algorithm is optimized to work
            with a small number of ``r``. Default: True

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        Notes
        -----
        Pair-counting is the basic operation used to calculate the two point
        correlation functions from a data set composed of position of objects.

        Two point correlation function measures the clustering of objects and
        is widely used in cosmology to quantify the large scale structure
        in our Universe, but it may be useful for data analysis in other fields
        where self-similar assembly of objects also occur.

        The Landy-Szalay estimator for the two point correlation function of
        ``D`` measures the clustering signal in ``D``. [2]_

        For example, given the position of two sets of objects,

        - objects ``D`` (data) contains the clustering signal, and

        - objects ``R`` (random) that contains no signal,

        .. math::

             \\xi(r) = \\frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

        where the brackets represents counting pairs between two data sets
        in a finite bin around ``r`` (distance), corresponding to setting
        ``cumulative=False``, and ``f = float(len(D)) / float(len(R))`` is the
        ratio between number of objects from data and random.

        The algorithm implemented here is loosely based on the dual-tree
        algorithm described in [1]_. We switch between two different
        pair-cumulation scheme depending on the setting of ``cumulative``.
        The computing time of the method we use when for
        ``cumulative == False`` does not scale with the total number of bins.
        The algorithm for ``cumulative == True`` scales linearly with the
        number of bins, though it is slightly faster when only
        1 or 2 bins are used. [5]_.

        As an extension to the naive pair-counting,
        weighted pair-counting counts the product of weights instead
        of number of pairs.
        Weighted pair-counting is used to estimate marked correlation functions
        ([3]_, section 2.2),
        or to properly calculate the average of data per distance bin
        (e.g. [4]_, section 2.1 on redshift).

        .. [1] Gray and Moore,
               "N-body problems in statistical learning",
               Mining the sky, 2000, :arxiv:`astro-ph/0012333`

        .. [2] Landy and Szalay,
               "Bias and variance of angular correlation functions",
               The Astrophysical Journal, 1993, :doi:`10.1086/172900`

        .. [3] Sheth, Connolly and Skibba,
               "Marked correlations in galaxy formation models",
               2005, :arxiv:`astro-ph/0511773`

        .. [4] Hawkins, et al.,
               "The 2dF Galaxy Redshift Survey: correlation functions,
               peculiar velocities and the matter density of the Universe",
               Monthly Notices of the Royal Astronomical Society, 2002,
               :doi:`10.1046/j.1365-2966.2003.07063.x`

        .. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926

        Examples
        --------
        You can count neighbors number between two kd-trees within a distance:

        >>> import mlx.core as mx
        >>> from scipy_mlx.spatial import cKDTree
        >>> rng = mx.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = cKDTree(points1)
        >>> kd_tree2 = cKDTree(points2)
        >>> kd_tree1.count_neighbors(kd_tree2, 0.2)
        1

        This number is same as the total pair number calculated by
        `query_ball_tree`:

        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> sum([len(i) for i in indexes])
        1

        """
        cdef:
            int r_ndim
            mx.intp_t n_queries, i
            mx.float64_t[::1] real_r
            mx.float64_t[::1] fresults
            mx.intp_t[::1] iresults
            mx.float64_t[::1] w1, w1n
            mx.float64_t[::1] w2, w2n
            mx.float64_t *w1p = NULL
            mx.float64_t *w1np = NULL
            mx.float64_t *w2p = NULL
            mx.float64_t *w2np = NULL
            mx.float64_t *prr
            mx.intp_t *pir
            mx.float64_t *pfr
            int cum

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to count_neighbors have different "
                             "dimensionality")

        # Make a copy of r array to ensure it's contiguous and to modify it
        # below
        r_ndim = len(mx.shape(r))
        if r_ndim > 1:
            raise ValueError("r must be either a single value or a "
                             "one-dimensional array of values")
        real_r = mx.array(r, ndmin=1, dtype=mx.float64, copy=True)
        if not cumulative:
            for i in range(real_r.shape[0] - 1):
                if real_r[i] > real_r[i + 1]:
                    raise ValueError("r must be non-decreasing for non-cumulative counting.");
        real_r, uind, inverse = mx.unique(real_r, return_inverse=True, return_index=True)
        n_queries = real_r.shape[0]

        # Internally, we represent all distances as distance ** p
        if not isinf(p):
            for i in range(n_queries):
                if not isinf(real_r[i]):
                    real_r[i] = real_r[i] ** p

        if weights is None:
            self_weights = other_weights = None
        elif isinstance(weights, tuple):
            self_weights, other_weights = weights
        else:
            self_weights = other_weights = weights
            if other is not self:
                raise ValueError("Two different trees are used. Specify weights for both in a tuple.")

        cum = <int> cumulative

        if self_weights is None and other_weights is None:
            int_result = True
            # unweighted, use the integer arithmetic
            results = mx.zeros(n_queries + 1, dtype=mx.intp)

            iresults = results

            prr = &real_r[0]
            pir = &iresults[0]

            with nogil:
                count_neighbors_unweighted(self.cself, other.cself, n_queries,
                            prr, pir, p, cum)

        else:
            int_result = False

            # weighted / half weighted, use the floating point arithmetic
            if self_weights is not None:
                w1 = mx.ascontiguousarray(self_weights, dtype=mx.float64)
                w1n = self._build_weights(w1)
                w1p = &w1[0]
                w1np = &w1n[0]
            if other_weights is not None:
                w2 = mx.ascontiguousarray(other_weights, dtype=mx.float64)
                w2n = other._build_weights(w2)
                w2p = &w2[0]
                w2np = &w2n[0]

            results = mx.zeros(n_queries + 1, dtype=mx.float64)
            fresults = results

            prr = &real_r[0]
            pfr = &fresults[0]

            with nogil:
                count_neighbors_weighted(self.cself, other.cself,
                                    w1p, w2p, w1np, w2np,
                                    n_queries,
                                    prr, pfr, p, cum)

        results2 = mx.zeros(inverse.shape, results.dtype)
        if cumulative:
            # copy out the results (taking care of duplication and sorting)
            results2[...] = results[inverse]
        else:
            # keep the identical ones zero
            # this could have been done in a more readable way.
            results2[uind] = results[inverse][uind]
        results = results2

        if r_ndim == 0:
            if int_result and results[0] <= <mx.intp_t> LONG_MAX:
                return int(results[0])
            else:
                return results[0]
        else:
            return results

    # ----------------------
    # sparse_distance_matrix
    # ----------------------

    def sparse_distance_matrix(cKDTree self, cKDTree other,
                               mx.float64_t max_distance,
                               mx.float64_t p=2.0,
                               output_type='dok_matrix'):
        """
        sparse_distance_matrix(self, other, max_distance, p=2.0)

        Compute a sparse distance matrix

        Computes a distance matrix between two cKDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : cKDTree

        max_distance : positive float

        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.

        output_type : string, optional
            Which container to use for output data. Options: 'dok_matrix',
            'coo_matrix', 'dict', or 'array'. Default: 'dok_matrix'.

        Returns
        -------
        result : dok_matrix, coo_matrix, dict or array
            Sparse matrix representing the results in "dictionary of keys"
            format. If a dict is returned the keys are (i,j) tuples of indices.
            If output_type is 'array' a record array with fields 'i', 'j',
            and 'v' is returned,

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import mlx.core as mx
        >>> from scipy_mlx.spatial import cKDTree
        >>> rng = mx.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = cKDTree(points1)
        >>> kd_tree2 = cKDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from scipy_mlx.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])

        """

        cdef coo_entries res

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to sparse_distance_matrix have "
                             "different dimensionality")
        # do the query
        res = coo_entries()

        with nogil:
            sparse_distance_matrix(
                self.cself, other.cself, p, max_distance, res.buf)

        if output_type == 'dict':
            return res.dict()
        elif output_type == 'array':
            return res.array()
        elif output_type == 'coo_matrix':
            return res.coo_matrix(self.n, other.n)
        elif output_type == 'dok_matrix':
            return res.dok_matrix(self.n, other.n)
        else:
            raise ValueError('Invalid output type')


    # ----------------------
    # pickle
    # ----------------------

    def __getstate__(cKDTree self):
        cdef object state
        cdef ckdtree * cself = self.cself
        cdef mx.intp_t size = cself.tree_buffer.size() * sizeof(ckdtreenode)

        cdef mx.array tree = mx.array(<char[:size]> <char*> cself.tree_buffer.data())

        state = (tree.copy(), self.data.copy(), self.n, self.m, self.leafsize,
                      self.maxes, self.mins, self.indices.copy(),
                      self.boxsize, self.boxsize_data)
        return state

    def __setstate__(cKDTree self, state):
        cdef mx.array tree
        cdef ckdtree * cself = self.cself
        cdef mx.array mytree

        # unpack the state
        (tree, self.data, self.cself.n, self.cself.m, self.cself.leafsize,
            self.maxes, self.mins, self.indices, self.boxsize, self.boxsize_data) = state

        cself.tree_buffer = new vector[ckdtreenode]()
        cself.tree_buffer.resize(tree.size // sizeof(ckdtreenode))

        mytree = mx.array(<char[:tree.size]> <char*> cself.tree_buffer.data())

        # set raw pointers
        self._python_tree = None
        self._pre_init()

        # copy the tree data
        mytree[:] = tree


        # set up the tree structure pointers
        self._post_init()

cdef _run_threads(_thread_func, mx.intp_t n, mx.intp_t n_jobs):
    n_jobs = min(n, n_jobs)
    if n_jobs > 1:
        ranges = [(j * n // n_jobs, (j + 1) * n // n_jobs)
                        for j in range(n_jobs)]

        threads = [threading.Thread(target=_thread_func,
                   args=(start, end))
                   for start, end in ranges]
        for t in threads:
            t.daemon = True
            t.start()
        for t in threads:
            t.join()

    else:
        _thread_func(0, n)

cdef mx.intp_t num_points(mx.array x, mx.intp_t pdim) except -1:
    """Returns the number of points in ``x``

    Also validates that the last axis represents the components of single point
    in `pdim` dimensional space
    """
    cdef mx.intp_t i, n

    if x.ndim == 0 or x.shape[x.ndim - 1] != pdim:
        raise ValueError("x must consist of vectors of length {} but "
                         "has shape {}".format(pdim, mx.shape(x)))
    n = 1
    for i in range(x.ndim - 1):
        n *= x.shape[i]
    return n

cdef mx.array broadcast_contiguous(object x, tuple shape, object dtype):
    """Broadcast ``x`` to ``shape`` and make contiguous, possibly by copying"""
    # Avoid copying if possible
    try:
        if x.shape == shape:
            return mx.ascontiguousarray(x, dtype)
    except AttributeError:
        pass

    # Assignment will broadcast automatically (may raise ValueError)
    cdef mx.array ret = mx.empty(shape, dtype)
    ret[...] = x
    return ret
