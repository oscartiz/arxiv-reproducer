from types import SimpleNamespace

from arxiv_reproducer.costs import Usage, estimate_cost_usd


def usage_msg(**kwargs):
    defaults = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestUsageAccumulation:
    def test_sums_across_messages(self):
        usage = Usage()
        usage.add(usage_msg(input_tokens=100, output_tokens=10))
        usage.add(usage_msg(input_tokens=200, output_tokens=20, cache_read_input_tokens=5000))
        assert usage.input_tokens == 300
        assert usage.output_tokens == 30
        assert usage.cache_read_input_tokens == 5000
        assert usage.requests == 2

    def test_tolerates_none_fields_and_missing_attrs(self):
        usage = Usage()
        usage.add(SimpleNamespace(input_tokens=None, output_tokens=7))  # no cache attrs
        assert usage.input_tokens == 0
        assert usage.output_tokens == 7

    def test_ignores_none_usage(self):
        usage = Usage()
        usage.add(None)
        assert usage.requests == 0

    def test_as_dict_shape(self):
        usage = Usage(input_tokens=1, output_tokens=2)
        assert usage.as_dict() == {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "api_requests": 0,
        }


class TestCostEstimate:
    def test_opus_pricing_math(self):
        # 100k in @$5 + 50k out @$25 + 1M cache-read @$0.50 + 200k cache-write @$6.25
        usage = Usage(
            input_tokens=100_000,
            output_tokens=50_000,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=200_000,
        )
        assert estimate_cost_usd(usage, "claude-opus-4-8") == 0.5 + 1.25 + 0.5 + 1.25

    def test_pure_input_run(self):
        usage = Usage(input_tokens=1_000_000)
        assert estimate_cost_usd(usage, "claude-opus-4-8") == 5.0

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd(Usage(input_tokens=100), "claude-mystery-9") is None
