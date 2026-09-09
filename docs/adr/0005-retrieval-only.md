# ADR 0005: No answer generation in the library

Status: accepted

## Context

Most RAG projects end with a call to an LLM. CRAG ends with packed context and
citations.

## Decision

The library never calls a generation model. `/v1/context` returns the text an
agent should put in its prompt, the citations backing it, and a quality label.
What to do with all three belongs to the caller.

## Why

Generation is the part every caller wants to control — model, prompt, tools,
streaming, refusal behaviour, cost ceiling. A library that owns it forces its
choices on the caller and becomes a wrapper around someone else's API. A library
that stops short composes with whatever the caller already runs.

It also makes the project measurable. Retrieval quality is testable against
public benchmarks with established metrics. End-to-end answer quality needs an
LLM judge, which is expensive, noisy, and mostly measures the generator. Every
number in [benchmarks.md](../benchmarks.md) is about the part this code controls.

The `retrieval_quality` label exists for the same reason: rather than deciding
what to do when retrieval turns up nothing, CRAG reports that it did and lets
the agent decide whether to answer from parametric knowledge, ask a follow-up,
or refuse.

## Cost

There is no single call that goes from question to answer, so every consumer
writes the generation step. `examples/agent_client.py` and
`demo_corpus/sample_code.py` show what it costs: a prompt template and one API
call.
