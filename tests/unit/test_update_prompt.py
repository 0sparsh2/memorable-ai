from memorable_ai.core.update import UPDATE_PROMPT


def test_update_prompt_format_no_keyerror():
    """JSON braces in template must not break str.format."""
    prompt = UPDATE_PROMPT.format(fact="I like Python", similar="(none)")
    assert "I like Python" in prompt
    assert '"operation"' in prompt
