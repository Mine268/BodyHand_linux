#pragma once

// Standard POSIX headers for sockets
#include <sys/socket.h>  // For socket(), bind(), listen(), accept(), etc.
#include <netinet/in.h>  // For sockaddr_in, IPPROTO_TCP, etc.
#include <arpa/inet.h>   // For inet_ntop(), etc.
#include <netdb.h>       // For getaddrinfo(), freeaddrinfo(), addrinfo
#include <unistd.h>      // For close()
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
#include <cstring>       // For strerror()

// Typedef for consistency (optional, as SOCKET is just an int on POSIX)
// Winsock uses SOCKET which is unsigned, POSIX uses int.
// We'll use int for POSIX compatibility.
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)

// Function declarations remain largely the same
bool InitSockets(); // Renamed from InitWinsock for clarity
void CleanupSockets(); // Renamed from CleanupWinsock

SOCKET CreateListeningSocket(const char* port, int backlog = SOMAXCONN);
SOCKET AcceptClient(SOCKET listenSock);
void CloseSocket(SOCKET s);

bool SendAll(SOCKET s, const void* buf, size_t len); // Changed len type to size_t
bool SendTextLine(SOCKET s, const std::string& line);

bool GetPeerString(SOCKET s, std::string& out);

void ServeClientWithCounter(SOCKET clientSock, int start = 0, int intervalSec = 1);
void ServeForever(const char* port);