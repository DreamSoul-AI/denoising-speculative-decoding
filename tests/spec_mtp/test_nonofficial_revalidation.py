"""The original fixed-history helper lineage remains a byte comparison."""







def test_copied_untimed_and_argmax_helpers_are_original_bytes(historical_file):
    for name in ('untimed_bf16.py', 'argmax_audit.py'):
        assert historical_file('studies/esp_table1_nonofficial_revalidation/' + name).read_bytes() == historical_file('studies/esp_table1_bf16_full_argmax/' + name).read_bytes()
