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
