# test/ - E2E/통합 테스트

Gateway 전체를 대상으로 하는 통합 테스트와 E2E 테스트.

- `fixtures/` - 테스트 픽스처 (설정 파일, 목 데이터)
- `helpers/` - 테스트 헬퍼 유틸리티
- `mocks/` - 목 객체/서비스
- `setup.ts` - 테스트 환경 설정
- `test-env.ts` - 테스트 환경변수 관리
- `gateway.multi.e2e.test.ts` - 멀티 에이전트 게이트웨이 E2E 테스트

실행: `pnpm test` (단위), `pnpm test:e2e` (E2E), `pnpm test:live` (라이브 모델)
