from songe.steps import digest, extract, split_for


def row(i, messages, action, tools=None, **kwargs):
    return {
        "id": i,
        "path": "/v1/chat/completions",
        "request": {"model": "songe/test", "messages": messages, "tools": tools or []},
        "response": {"choices": [{"message": action}]},
        "status": 200,
        "complete": True,
        **kwargs,
    }


def test_opencode_parallel_tools_hindsight():
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "parameters": {"type": "object"}},
        }
    ]
    c = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Read both files."},
    ]
    a = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"a.py"}'},
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"b.py"}'},
            },
        ],
    }
    results = [
        {"role": "tool", "tool_call_id": "call_a", "content": "file not found"},
        {"role": "tool", "tool_call_id": "call_b", "content": "print(1)"},
    ]
    # Content null is replayed as empty by some client transports.
    replay = a | {"content": ""}
    out = extract(
        [
            row(1, c, a, tools),
            row(
                2,
                c + [replay] + results,
                {"role": "assistant", "content": "a.py is missing."},
                tools,
            ),
        ]
    )
    assert len(out) == 2
    assert out[0].hindsight == results
    assert out[0].tools == tools
    assert out[0].conversation == out[1].conversation
    assert out[0].split == out[1].split


def test_pi_text_blocks_and_next_user():
    c = [
        {"role": "user", "content": [{"type": "text", "text": "What is two plus two?"}]}
    ]
    a = {"role": "assistant", "content": "Five", "reasoning_content": "thinking"}
    replay = [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "Five"},
    ]
    h = {"role": "user", "content": "That should be four."}
    steps = extract(
        [row(1, c, a), row(2, replay + [h], {"role": "assistant", "content": "Four"})]
    )
    assert steps[0].hindsight == [h]
    assert not steps[1].hindsight
    assert steps[0].conversation == steps[1].conversation


def test_branching_and_stable_splits():
    c = [{"role": "user", "content": "Start"}]
    a = {"role": "assistant", "content": "Hello"}
    logs = [
        row(1, c, a),
        row(2, c + [a, {"role": "user", "content": "Branch A"}], a),
        row(3, c + [a, {"role": "user", "content": "Branch B"}], a),
    ]
    # One session is cut at each new instruction: three instructions, three
    # parts, however the session branched.
    assert len({s.conversation for s in extract(logs)}) == 3
    assert [s.split for s in extract(logs)] == [s.split for s in extract(logs)]
    # Splits wait until enough data exists; Store.assign_splits settles them.
    assert {split_for(digest(str(i))) for i in range(10)} == {"pending"}


def test_a_session_is_cut_at_each_new_instruction():
    """People stay in one session all day: counted whole, it never became the
    three conversations a first update needs (live: 41 examples, 2 sessions)."""
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "run", "arguments": "{}"}}],
    }
    done = {"role": "assistant", "content": "Done"}
    first = [{"role": "user", "content": "Build the page"}]
    result = {"role": "tool", "tool_call_id": "1", "content": "ok"}
    fix = {"role": "user", "content": "That is wrong, fix the title"}
    second = {"role": "user", "content": "Now add a footer"}
    steps = extract(
        [
            row(1, first, call),
            row(2, first + [call, result], done),
            row(3, first + [call, result, done, fix], done),
            row(4, first + [call, result, done, fix, done, second], done),
        ]
    )
    parts = [s.conversation for s in steps]
    # The tool call, its follow-up and the correction are one piece of work.
    assert parts[0] == parts[1] == parts[2]
    # A new instruction starts the next one.
    assert parts[3] != parts[0]
    assert all(s.conversation == s.task for s in steps)


def test_failed_partial_multichoice_other_model_and_compaction_excluded():
    c = [{"role": "user", "content": "Hi"}]
    a = {"role": "assistant", "content": "Hello"}
    logs = [
        row(1, c, a, status=500),
        row(2, c, a, complete=False),
        row(3, c, a, response={"choices": [{"message": a}, {"message": a}]}),
        row(4, [a, *c], a),
    ]
    assert not extract(logs)
    assert not extract([row(5, c, a)], "songe/another")


def test_deduplicates_retries_and_preserves_tool_arguments():
    c = [{"role": "user", "content": "Run tests"}]
    a = {"role": "assistant", "content": "Okay"}
    assert len(extract([row(1, c, a), row(2, c, a)])) == 1
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    assert digest({"name": "read", "arguments": '{"b":1,"a":2}'}) == digest(
        {"name": "read", "arguments": {"a": 2, "b": 1}}
    )
