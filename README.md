# Jinuk Huh — Research, CV & Blog

Jekyll 기반 한국어·영어 연구자 홈페이지. GitHub Pages의 기존 배포 방식을 유지합니다.

## 콘텐츠 수정

- `_data/academic.json`: 두 언어의 소개·소속·메뉴 문구. 확인된 박사과정 정보만 기재하며 입학 연도·이전 학위는 자료 확보 후 추가합니다.
- `_data/publications.json`: 논문 서지와 DOI. 영문 이름은 사용자 확인에 따라 `Jinuk Huh`로 통일합니다. GitHub 계정은 `HeoJinuk`입니다.
- `assets/publications/advanced-dyna-q.bib`: 다운로드용 BibTeX. 논문 데이터를 바꾸면 함께 수정합니다.
- `_posts/YYYY-MM-DD-title.markdown`: 블로그 글. `title`, `lang`, `category`를 YAML front matter에 지정합니다. 기존 글은 한국어 원문이며 영어 목록에서도 언어를 표시합니다.
- `_layouts/academic*.html`, `_includes/academic-*.html`: 레이아웃과 공통 구성.
- `assets/css/academic.css`: 반응형 및 CV 인쇄 스타일.

## 경로

한국어 `/`, `/cv/`, `/blog/`; 영어 `/en/`, `/en/cv/`, `/en/blog/`.
CV 페이지의 인쇄 버튼으로 브라우저에서 PDF로 저장할 수 있습니다.
기존 `/tabs/` 페이지는 새 페이지로 연결하고, 글 permalink `/posts/:path`는 유지합니다.

## 로컬 실행

기존 Gemfile의 GitHub Pages 의존성을 설치한 뒤 실행합니다.

```sh
bundle install
bundle exec jekyll serve
```

배포 전 `bundle exec jekyll build`와 데스크톱·모바일 화면, 언어 전환, 글 이미지, DOI/BibTeX, CV 인쇄를 확인합니다.

## 논문 출처

JICS 26(6), 93–100 (2025): https://doi.org/10.7472/jksii.2025.26.6.93
공식 목차: https://www.jics.or.kr/digital-library/publication?number=6&volume=26

미확인 경력·학위·수상·성과 수치는 추가하지 않았습니다. 사용하지 않는 기존 테마·샘플 리소스는 제거했습니다. 기존 라이선스는 보존합니다.

## 브라우저에서 글쓰기 — 이 컴퓨터

1. 저장소의 `블로그 글쓰기.command`를 더블클릭합니다. 편집기와 홈페이지 미리보기 서버가 함께 실행됩니다.
2. `http://127.0.0.1:4174/admin/`에서 로컬 저장소 연결 버튼을 누릅니다. 계정 로그인은 필요 없습니다.
3. **블로그 글 → 새 글**에서 제목·날짜·언어·분류와 본문을 입력합니다. 본문은 서식 편집/Markdown을 전환할 수 있고 이미지도 업로드할 수 있습니다.
4. 저장(편집기 버전에 따라 Publish로 표시)하면 `_posts`에 실제 파일을 씁니다. 이 버튼은 로컬 저장이며 GitHub 공개 발행이 아닙니다. 미리보기 `http://127.0.0.1:4174/blog/`를 새로고침하면 결과를 볼 수 있습니다.
5. 미공개로 보관하려면 **블로그 목록에 공개**를 끕니다. 파일은 편집기에서 다시 열 수 있지만 Jekyll 블로그 목록에는 나타나지 않습니다.
6. 공개 사이트 반영은 변경 파일 확인 → 커밋 → push → GitHub Pages 빌드 순서로 별도 진행합니다. 편집기는 Git 작업을 자동 수행하지 않습니다.
7. 종료할 때 실행 창에서 `Ctrl+C`를 누릅니다.

현재 설치한 컴퓨터에서는 Node·Jekyll 환경을 준비해 두었습니다. 새 컴퓨터에서는 Node.js 20 이상과 프로젝트의 Ruby/Bundler 환경을 준비하고 `pnpm install --frozen-lockfile`(또는 `npm install`), `bundle install`을 실행합니다. 이후 `node scripts/editor.mjs`로도 시작할 수 있습니다. `.local-editor/gems`가 있으면 로컬 검증용 Jekyll을 우선 사용하고, 없으면 기존 Gemfile의 `bundle exec jekyll`을 사용합니다. `.local-editor`, `node_modules`는 Git에 포함하지 않습니다.

- CMS: Decap CMS 3.16.3, 로컬 파일 서버: decap-server 3.11.6 (`MODE=fs`).
- 이미지 저장 위치: `assets/img/posts/uploads`.
- 글 삭제 메뉴는 비활성화했습니다. 기존 글의 제목을 바꿔도 파일명·기존 URL은 유지합니다.
- 본문 미리보기는 글 스타일과 Markdown을 보여줍니다. Liquid·수식의 최종 렌더링은 홈페이지 미리보기에서 확인합니다.
- 편집기 스크립트는 고정 버전의 jsDelivr CDN에서 로드하므로 인터넷 연결이 필요합니다.
- 서버는 `127.0.0.1`에만 열리고 편집기 포트(4174)에서 온 요청만 허용합니다. 공개 도메인에서는 편집기를 시작하지 않습니다.
- 현재 외부 OAuth나 공개 사이트 관리자 로그인은 연결하지 않았습니다. 이 컴퓨터의 브라우저에서만 작성합니다.

공식 문서: https://decapcms.org/docs/decap-proxy/ · https://decapcms.org/docs/jekyll/
