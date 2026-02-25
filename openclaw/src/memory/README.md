# src/memory/ - 벡터 메모리 시스템

sqlite-vec 기반 벡터 검색으로 에이전트의 장기 메모리 제공.
대화 내용을 벡터화하여 저장하고, 관련 컨텍스트를 검색합니다.

## 연동
- ← `src/agents/memory-search.ts` (에이전트가 메모리 검색)
- ← `extensions/memory-core/`, `extensions/memory-lancedb/` (메모리 백엔드)
