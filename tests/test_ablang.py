import ablang

# import numpy as np
# import numpy.testing as npt


def test_ablang():
    model = ablang.pretrained("light")
    model.freeze()

    seqs = [
        "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
    ]

    tokens = model.tokenizer(seqs, pad=True)

    assert tuple(tokens.shape) == (2, 123)

    # fmt: off
    expected_tokens = [
        [
            0, 6, 15, 10, 20, 15, 6, 7, 12, 13, 12, 20, 15, 10, 13, 12, 4, 7,
            20, 2, 20, 7, 11, 15, 14, 7, 12, 17, 8, 17, 7, 12, 18, 12, 1, 3, 19,
            15, 2, 10, 14, 13, 12, 4, 12, 20, 6, 19, 16, 14, 20, 16, 16, 18, 5,
            6, 7, 9, 4, 18, 18, 14, 5, 7, 15, 4, 12, 2, 17, 8, 16, 7, 2, 5, 9,
            7, 4, 9, 8, 20, 18, 20, 10, 1, 7, 7, 20, 2, 14, 6, 5, 8, 14, 15, 17,
            18, 11, 14, 4, 15, 4, 17, 18, 5, 13, 8, 14, 13, 9, 5, 18, 19, 12,
            10, 12, 8, 20, 15, 8, 15, 7, 7, 22,
        ],
        [
            0, 10, 15, 10, 20, 15, 10, 7, 12, 14, 6, 15, 4, 4, 13, 12, 14, 7,
            15, 4, 15, 7, 11, 4, 14, 7, 12, 18, 8, 17, 8, 7, 18, 12, 16, 7, 19,
            15, 2, 10, 14, 13, 12, 10, 12, 20, 6, 19, 1, 12, 19, 16, 7, 14, 18,
            9, 12, 9, 8, 9, 18, 14, 10, 4, 20, 10, 12, 2, 15, 8, 1, 8, 8, 5, 8,
            7, 8, 7, 8, 14, 18, 1, 6, 20, 2, 7, 20, 2, 7, 5, 5, 8, 14, 15, 18,
            18, 11, 14, 2, 15, 20, 12, 19, 12, 7, 1, 5, 15, 19, 12, 10, 12, 8,
            8, 15, 8, 15, 7, 7, 22, 21, 21, 21]
    ]
    # fmt: on

    assert tokens.tolist() == expected_tokens
