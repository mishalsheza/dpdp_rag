# Prompts

Every LLM prompt used by the system lives here as its own file (`<stage>_<purpose>.md`),
and code loads it at runtime. Prompts are never inlined in Python. Use `{placeholders}` for
values filled in at call time, and keep the prompt's expected inputs listed at the top of
the file.
