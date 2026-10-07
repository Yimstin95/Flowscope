# 배포 가이드 (Streamlit Community Cloud, 무료)

## 배포가 뭔가요?
지금 FlowScope는 **내 컴퓨터 안에서만** 열립니다. 배포는 이 앱을 **인터넷 주소**(예: `https://flowscope-xxxx.streamlit.app`)로
열어서, 면접관이 링크 하나만 눌러도 볼 수 있게 만드는 일입니다. 비유하면 집에서 만든 요리를 **푸드코트에 매대로 내는 것**입니다.
Streamlit Community Cloud는 무료 푸드코트이고, GitHub에 있는 코드를 가져가서 대신 실행해 줍니다.

## 먼저 알아둘 제한 (공식 문서·포럼 기준, 변동 가능)
| 항목 | 내용 |
|---|---|
| 비용 | 무료. 공개 앱은 개수 제한 없음 |
| 공개 범위 | **공개 저장소 = 공개 앱.** 코드가 누구에게나 보임 |
| 메모리 | 최대 약 2.7GB. 넘으면 앱이 멈춤 |
| 수면 | 12시간 동안 방문이 없으면 잠듦. 다음 방문자가 "깨우기"를 눌러야 함 |
| 파이썬 | 기본 3.12. 고급 설정에서 바꿀 수 있음 |

## 단계

### 1단계 · 올리기 전 점검 (제가 같이 확인)
- [ ] 테스트 통과: `pytest`
- [ ] **비밀이 없는지**: `.env`(키 파일)는 올라가지 않아야 함. `git check-ignore .env`로 확인
- [ ] **큰 파일이 없는지**: 395MB짜리 OMIP-024 원본은 올리지 않음(이미 제외됨). 작은 실제 샘플(4MB)과 평가 결과만 포함
- [ ] 평가 결과 파일(`data/eval/*.csv`)은 올림. 방문자는 이것을 보고 결과를 확인

### 2단계 · GitHub에 올리기
저장소는 이미 있습니다(`Yimstin95/Flowscope`). 변경한 파일을 올립니다.
```bash
git add -A
git commit -m "Add expert-graded evaluation of the Critic + free local-LLM provider"
git push
```
※ 올리기는 **밖으로 공개되는 행동**이라, 실행 전에 반드시 당신 허락을 받습니다.

### 3단계 · 앱 만들기 (당신이 직접, 약 5분)
1. https://share.streamlit.io 접속 → **GitHub로 로그인**
2. **Create app** → "Deploy a public app from GitHub" 선택
3. 입력:
   - Repository: `Yimstin95/Flowscope`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. **Advanced settings**에서
   - Python version: 3.12 또는 3.13
   - Secrets: **비워 둡니다** (아래 "중요" 참고)
5. **Deploy** 클릭 → 첫 빌드는 수 분 걸립니다(UMAP 등 무거운 패키지 설치).

### 4단계 · 확인
- 화면 오른쪽 아래 **Manage app**에서 로그를 볼 수 있습니다. 빌드가 실패하면 그 로그를 저에게 붙여 주세요.
- 직접 해볼 것: 번들 샘플이 열리는지 → 클러스터링이 끝나는지 → 평가 탭 결과가 보이는지
- 느리거나 "resource limits" 메시지가 나오면 메모리 한도 문제입니다. 샘플 이벤트 수를 줄이면 됩니다.

### 5단계 · 링크 쓰기
- README 맨 위에 "Live demo" 링크, LinkedIn·이력서에도 넣습니다.
- 면접·지원 **직전에 링크를 한 번 눌러서** 앱을 깨워 두세요(12시간 수면 때문).

## 중요: API 키는 Secrets에 넣지 않습니다
기존 README에는 "AI 리포트를 위해 키를 Secrets에 넣으라"고 되어 있지만, **권하지 않습니다.**
공개 앱에 키를 넣으면 누구나 AI 버튼을 눌러 **당신의 요금을 쓰게** 되기 때문입니다.
그래서 이번 배포는:
- 클러스터링·히트맵·이상 탐지는 **누구나 바로** 사용(무료, 키 불필요)
- AI 판단은 **미리 계산해 둔 평가 결과**를 보여줌(방문자가 호출하지 않으므로 비용 0원)
