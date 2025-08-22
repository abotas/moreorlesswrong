To orient read @README.md

- Never write defensive code.
  - Never use try/except clauses
  - It's way better to crash than to silently work around unexpected data
  - Almost never use kwargs with defaults. never have return types that can be {type} | None
  - Be opinionated and decisive. Don't implement a fn like this: def fetch_wikipedia_snapshots(page_name: str, dates: list[str | date]) -> None: that can defensively take multiple types of args. implement def fetch_wikipedia_snapshots(page_name: str, dates: list[date])
  - never add logic to support backwards compatibility
- be a ruthless editor. try very hard to write only the bare essential logic. 
- if your function args or return types are numerous or convoluted, scrutinize why and fix things. you should basically never be returning list[tuple[str, datetime]], for example.
- If one of my instructions does not make sense, do not implement it. Ask for clarification or explain why an instruction doesn't make sense
- If you're in the middle of a task and you realize something could use clarification, stop and ask for clarification or choose the best option (if one is obvious). Don't implement something that might work for either. 
- If you're creating a new file, before starting to code think critically about what interfaces make the most sense with the existing code base. Keep things as simple as possible.
- You may drive-by simplify if you notice opportunities, but only for true, small simplifications
- gpt5 is out do not switch me back to gpt4o gpt5 is better and cheaper
- do not test streamlit by launching it. it's ok to not test it unless you have a really good idea for how to do it easily without spending API credits or writing tons of code.
- if you're doing a no-op refactoring job for pure code simplification and get to the end, check that you did actually make the code simpler. if not, consider reverting your change and (maybe) trying again
