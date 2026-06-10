import pytest

from arxiv_reproducer.paper import parse_arxiv_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2301.12345", "2301.12345"),
        ("2301.12345v2", "2301.12345v2"),
        ("https://arxiv.org/abs/2301.12345", "2301.12345"),
        ("https://arxiv.org/pdf/2301.12345v1", "2301.12345v1"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("  2301.12345  ", "2301.12345"),
    ],
)
def test_parse_arxiv_id(raw, expected):
    assert parse_arxiv_id(raw) == expected


def test_parse_arxiv_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_arxiv_id("not a paper")
