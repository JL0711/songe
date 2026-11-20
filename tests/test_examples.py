"""Before/after example answers: tool calls in the forms models really write."""

from types import SimpleNamespace

from songe.sleep.evaluate import xml_call
from songe.sleep.examples import answer

TOOLS = [
    {"type": "function", "function": {"name": "glob", "parameters": {
        "type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"filePath": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["filePath"]}}},
]
TOKENIZER = SimpleNamespace(chat_template="<tool_call></tool_call>")


def test_xml_style_calls_parse_with_schema_types():
    call = xml_call("\n<function=read>\n<parameter=filePath>\na.py\n</parameter>\n"
                    "<parameter=limit>\n20\n</parameter>\n</function>\n", TOOLS)
    assert call == {"name": "read", "arguments": {"filePath": "a.py", "limit": 20}}
    assert xml_call('{"name": "read"}', TOOLS) is None


def test_an_answer_is_valid_when_its_call_parses_and_names_a_known_tool():
    good = "Looking.</think><tool_call>\n<function=glob>\n<parameter=pattern>\n**/*.py\n</parameter>\n</function>\n</tool_call>"
    parsed = answer(good, TOKENIZER, TOOLS)
    assert parsed["valid"] and parsed["tool_calls"] == [{"name": "glob", "arguments": {"pattern": "**/*.py"}}]
    # Cut off at the token limit, or a tool the agent does not have: invalid.
    assert not answer(good[:-12], TOKENIZER, TOOLS)["valid"]
    assert not answer(good.replace("glob", "list_dir"), TOKENIZER, TOOLS)["valid"]
    # The JSON form still works, and plain text is valid when not empty.
    json_form = '<tool_call>{"name": "read", "arguments": {"filePath": "a.py"}}</tool_call>'
    assert answer(json_form, TOKENIZER, TOOLS)["valid"]
    assert answer("Done: the tests pass.", TOKENIZER, TOOLS)["valid"]
    assert not answer("", TOKENIZER, TOOLS)["valid"]
