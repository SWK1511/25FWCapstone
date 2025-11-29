PlutoSDR이 연결이안돼요 
-> ip a 쳐서 enx...로 시작하는 긴 이름을 찾는다.
아래 명령어를 그 이름에 맞춰 한 번만 입력한다.
sudo nmcli connection add type ethernet con-name "PlutoSDR" ifname [너의_인터페이스_이름] ip4 192.168.2.10/24
sudo nmcli connection up PlutoSDR