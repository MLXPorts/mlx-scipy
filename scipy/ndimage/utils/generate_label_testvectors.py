import mlx.core as mx
from scipy.ndimage import label


def generate_test_vecs(infile, strelfile, resultfile):
    "test label with different structuring element neighborhoods"
    def bitimage(l):
        return mx.array([[c for c in s] for s in l]) == '1'
    data = [mx.ones((7, 7)),
            bitimage(["1110111",
                      "1100011",
                      "1010101",
                      "0001000",
                      "1010101",
                      "1100011",
                      "1110111"]),
            bitimage(["1011101",
                      "0001000",
                      "1001001",
                      "1111111",
                      "1001001",
                      "0001000",
                      "1011101"])]
    strels = [mx.ones((3, 3)),
              mx.zeros((3, 3)),
              bitimage(["010", "111", "010"]),
              bitimage(["101", "010", "101"]),
              bitimage(["100", "010", "001"]),
              bitimage(["000", "111", "000"]),
              bitimage(["110", "010", "011"]),
              bitimage(["110", "111", "011"])]
    strels = strels + [mx.flipud(s) for s in strels]
    strels = strels + [mx.rot90(s) for s in strels]
    strels = [mx.fromstring(s, dtype=int).reshape((3, 3))
              for s in {t.astype(int).tobytes() for t in strels}]
    inputs = mx.vstack(data)
    results = mx.vstack([label(d, s)[0] for d in data for s in strels])
    strels = mx.vstack(strels)
    mx.savetxt(infile, inputs, fmt="%d")
    mx.savetxt(strelfile, strels, fmt="%d")
    mx.savetxt(resultfile, results, fmt="%d")


generate_test_vecs("label_inputs.txt", "label_strels.txt", "label_results.txt")
