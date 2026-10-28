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
