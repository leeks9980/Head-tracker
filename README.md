# 🏎️ F1 25 Multi-PC Head & Gaze Tracking System

> **MediaPipe Face Mesh**와 **OpenTrack (FreePIE UDP)**을 활용한 멀티 PC 기반 실시간 헤드 및 아이 트래킹 연동 프로젝트입니다. 웹캠이 장착된 보조 기기(노트북 등)에서 얼굴/시선 추적 연산을 수행하고, 네트워크(UDP)를 통해 게임이 실행 중인 메인 PC로 60Hz 고정 신호를 전송하여 레이싱 시뮬레이터(*F1 25*)의 시점을 제어합니다.

---

## 🛠️ System Architecture

```text
[ 보조 PC (웹캠 연동) ]                    [ 메인 PC (게임 플레이) ]
+-------------------------+      UDP       +--------------------+      Shared Memory    +-----------+
| Python + MediaPipe      | -------------> | OpenTrack          | -------------------> | F1 25     |
| (60Hz Threaded Sender)  |   (Port 5555)  | (FreePIE Receiver) |                      | (Game)    |
+-------------------------+                +--------------------+                      +-----------+


### 2. `QUICKSTART.md` (실행 및 설정 가이드)

```markdown
# ⚡ Quick Start Guide

이 문서는 멀티 PC 환경에서 트래커와 OpenTrack, 그리고 *F1 25*를 연동하기 위한 절차를 다룬다.

## 1. 사전 준비
* 두 대의 PC가 **동일한 네트워크(공유기)**에 연결되어 있어야 한다.
* 메인 PC(게임 실행)의 방화벽에서 **UDP 5555 포트**가 개방되어 있어야 한다.

## 2. 메인 PC 설정 (OpenTrack)
1. **OpenTrack**을 실행한다.
2. 상단 입력(Input) 드롭다운에서 **[FreePIE UDP receiver]**를 선택하고 설정(포트 `5555`)을 확인한다.
3. 상단 출력(Output) 드롭다운에서 **[Freetrack 2.0 Enhanced]**를 선택한다.
4. *F1 25* 게임을 실행하고 설정에서 헤드 트래킹/TrackIR 연동이 활성화되어 있는지 확인한다.

## 3. 보조 PC 설정 (Python Script)
1. 웹캠이 연결된 보조 PC에서 가상환경을 활성화하고 라이브러리를 설치한다.
2. 파이썬 코드 내의 `OPENTRACK_IP` 변수를 **메인 PC의 로컬 IP 주소**로 변경한다.
3. 스크립트를 실행한다.

```bash
python peristalsis.py
