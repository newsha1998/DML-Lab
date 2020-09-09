# How to connect to the server?
First, connect to your Sharif VPN. then:
```sh
$ ssh drrohban@192.168.207.154
```

Before connect to the VPN, check that the ip does not change.
```sh
$ tmux attach -t vpn
$ check-vpn
```
After 1 min, if you are still connected to the server, the VPN is working properly. Thus, you can connect to the VPN.

Connect to the VPN:
```sh
$ tmux attach -t vpn
$ vpn
```

Close tmux:
```sh
1. Ctrl + B
2. D
```

Disconnect VPN:
```sh
$ tmux attach -t vpn
$ sudo pkill openconnect
```