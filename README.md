# U-CCCP CUDA Project

CUDA 프로그래밍을 위한 프로젝트입니다.

## 환경 정보
- CUDA Version: 12.9
- GPU: Tesla P4
- OS: Debian 12 (bookworm)

## 프로젝트 구조
```
U-CCCP/
├── src/           # CUDA 소스 파일들
├── include/       # 헤더 파일들
├── bin/           # 컴파일된 실행 파일들
├── build/         # 오브젝트 파일들
├── Makefile       # 빌드 설정
└── README.md      # 프로젝트 설명
```

## 빌드 및 실행
```bash
# 컴파일
make

# 실행
make run

# 정리
make clean
```
