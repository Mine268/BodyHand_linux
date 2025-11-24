#include "tcp_server.h"
#include <cerrno>  // For errno
#include <iostream> // Optional: for more detailed error messages if needed

bool InitSockets() {
    // On POSIX systems, no explicit initialization like WSAStartup is required.
    // Sockets are based directly on file descriptors.
    return true; // Assume success for POSIX
}

void CleanupSockets() {
    // On POSIX systems, no global cleanup like WSACleanup is needed.
    // Resources are managed per process/file descriptor.
}

SOCKET CreateListeningSocket(const char* port, int backlog) {
    struct addrinfo hints{}; // Value-initialize to zero
    struct addrinfo *result = nullptr, *rp;

    // Prepare hints structure
    hints.ai_family = AF_INET;       // Use AF_INET6 for IPv6 or AF_UNSPEC for either
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP; // Often implied by SOCK_STREAM, but good practice
    hints.ai_flags = AI_PASSIVE;     // Fill in my IP address

    // Resolve the address and port
    int s = getaddrinfo(nullptr, port, &hints, &result);
    if (s != 0) {
        std::fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        return INVALID_SOCKET;
    }

    SOCKET listen_sock = INVALID_SOCKET;
    // Loop through results and bind to the first possible
    for (rp = result; rp != nullptr; rp = rp->ai_next) {
        listen_sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (listen_sock == INVALID_SOCKET) {
            perror("socket"); // Prints error message based on errno
            continue; // Try next address
        }

        // Enable address reuse (helpful for quick restarts)
        int yes = 1;
        if (setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
             perror("setsockopt SO_REUSEADDR");
             close(listen_sock);
             listen_sock = INVALID_SOCKET;
             continue;
        }

        // Bind the socket
        if (bind(listen_sock, rp->ai_addr, rp->ai_addrlen) == SOCKET_ERROR) {
            perror("bind");
            close(listen_sock);
            listen_sock = INVALID_SOCKET;
            continue;
        }

        // Successfully bound, break out of loop
        break;
    }

    freeaddrinfo(result); // No longer needed

    if (listen_sock == INVALID_SOCKET) {
         fprintf(stderr, "Could not bind to any address\n");
         return INVALID_SOCKET;
    }

    // Start listening
    if (listen(listen_sock, backlog) == SOCKET_ERROR) {
        perror("listen");
        close(listen_sock);
        return INVALID_SOCKET;
    }

    return listen_sock;
}


SOCKET AcceptClient(SOCKET listenSock) {
    std::printf("Waiting for a client to connect...\n");
    SOCKET client_sock = accept(listenSock, nullptr, nullptr); // Blocking call
    if (client_sock == INVALID_SOCKET) {
        perror("accept"); // Use perror for POSIX errors
    }
    return client_sock;
}

void CloseSocket(SOCKET s) {
    if (s != INVALID_SOCKET) {
        // shutdown is optional here if you're just closing, but good practice
        if (shutdown(s, SHUT_RDWR) == SOCKET_ERROR) {
             if(errno != ENOTCONN) { // Ignore error if not connected
                 perror("shutdown");
             }
        }
        close(s); // Use close() instead of closesocket()
    }
}

bool SendAll(SOCKET s, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t sentTotal = 0;
    while (sentTotal < len) {
        ssize_t sent = send(s, p + sentTotal, len - sentTotal, 0);
        if (sent == SOCKET_ERROR) {
            perror("send");
            return false;
        }
        sentTotal += sent;
    }
    return true;
}

bool SendTextLine(SOCKET s, const std::string& line) {
    std::string withNL = line;
    if (withNL.empty() || withNL.back() != '\n') withNL.push_back('\n');
    return SendAll(s, withNL.c_str(), withNL.size()); // size_t is correct now
}

bool GetPeerString(SOCKET s, std::string& out) {
    struct sockaddr_storage ss{};
    socklen_t len = sizeof(ss);

    if (getpeername(s, reinterpret_cast<struct sockaddr*>(&ss), &len) != 0) {
        perror("getpeername");
        return false;
    }

    char host[NI_MAXHOST] = {'\0'};
    char serv[NI_MAXSERV] = {'\0'};

    int res = getnameinfo(reinterpret_cast<struct sockaddr*>(&ss), len,
                          host, sizeof(host),
                          serv, sizeof(serv),
                          NI_NUMERICHOST | NI_NUMERICSERV);

    if (res != 0) {
        fprintf(stderr, "getnameinfo: %s\n", gai_strerror(res));
        return false;
    }

    out = std::string(host) + ":" + std::string(serv);
    return true;
}


void ServeClientWithCounter(SOCKET clientSock, int start, int intervalSec) {
    std::string peer;
    if (GetPeerString(clientSock, peer)) {
        std::printf("Client connected from %s\n", peer.c_str());
    } else {
        std::printf("Client connected (failed to get peer info)\n");
    }

    int counter = start;
    while (true) {
        std::string msg = "Hello from server, count=" + std::to_string(counter++);
        if (!SendTextLine(clientSock, msg)) {
            // Check errno or handle specific conditions if needed
            std::printf("send failed or peer closed.\n");
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(intervalSec));
    }

    CloseSocket(clientSock);
    std::printf("Client disconnected.\n");
}

void ServeForever(const char* port) {
    if (!InitSockets()) {
        return; // Should rarely fail on POSIX
    }

    SOCKET ls = CreateListeningSocket(port);
    if (ls == INVALID_SOCKET) {
        // Error already printed in CreateListeningSocket
        CleanupSockets(); // Still call cleanup even if init seemed okay
        return;
    }
    std::printf("Server started. Listening on port %s ...\n", port);

    // Permanent loop: serve one client, then wait for the next
    while (true) {
        SOCKET cs = AcceptClient(ls);
        if (cs == INVALID_SOCKET) {
            // Brief pause before retrying accept on error
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        // Serve the client. This blocks until the client disconnects.
        // To handle multiple clients simultaneously, you'd need threads/processes here.
        ServeClientWithCounter(cs); // Replace with your own logic
        // Note: ServeClientWithCounter calls CloseSocket internally.
    }

    // Code below usually unreachable in this loop structure
    CloseSocket(ls);
    CleanupSockets();
}