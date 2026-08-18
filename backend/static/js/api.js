const Api = {
    getCsrfToken() {
        const match = document.cookie.match(/csrftoken=([^;]+)/);
        return match ? decodeURIComponent(match[1]) : "";
    },

    async request(url, options = {}) {
        const headers = {
            Accept: "application/json",
            ...(options.headers || {}),
        };

        if (options.body && !(options.body instanceof FormData)) {
            headers["Content-Type"] = "application/json";
        }

        const csrfToken = this.getCsrfToken();
        if (csrfToken) {
            headers["X-CSRFToken"] = csrfToken;
        }

        const response = await fetch(url, {
            ...options,
            headers,
        });

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) {
            const message =
                data?.detail ||
                data?.file?.[0] ||
                data?.question?.[0] ||
                data?.document_id?.[0] ||
                "Something went wrong. Please try again.";

            throw new Error(message);
        }

        return data;
    },

    uploadDocument(file) {
        const formData = new FormData();
        formData.append("file", file);

        return this.request("/api/documents/upload/", {
            method: "POST",
            body: formData,
        });
    },

    processDocument(documentId) {
        return this.request(`/api/documents/${documentId}/process/`, {
            method: "POST",
        });
    },

    resetConversation() {
        return this.request("/api/chat/new/", {
            method: "POST",
        });
    },

    askQuestion(documentId, question) {
        return this.request("/api/chat/ask/", {
            method: "POST",
            body: JSON.stringify({
                document_id: documentId,
                question,
            }),
        });
    },

    /**
     * Streaming version of askQuestion.
     *
     * Sends a POST to /api/chat/ask/stream/ and reads the Server-Sent Events
     * response token by token.
     *
     * @param {string} question
     * @param {function} onToken   - called with each token string as it arrives
     * @param {function} onDone    - called once with {answer, sources} when stream ends
     * @param {function} onError   - called with an error message string on failure
     */
    async askQuestionStream(question, onToken, onDone, onError) {
        const headers = {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
        };

        const csrfToken = this.getCsrfToken();
        if (csrfToken) {
            headers["X-CSRFToken"] = csrfToken;
        }

        let response;
        try {
            response = await fetch("/api/chat/ask/stream/", {
                method: "POST",
                headers,
                body: JSON.stringify({ question }),
            });
        } catch (err) {
            onError("Network error. Please check your connection.");
            return;
        }

        if (!response.ok) {
            let message = "Something went wrong. Please try again.";
            try {
                const data = await response.json();
                message = data?.detail || message;
            } catch { /* ignore */ }
            onError(message);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        let settled = false;

        const settle = (callback, argument) => {
            if (settled) return;
            settled = true;
            callback(argument);
        };

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // SSE lines look like "data: {...}\n\n" — split on double newline.
                const parts = buffer.split("\n\n");
                buffer = parts.pop(); // keep the incomplete trailing part

                for (const part of parts) {
                    const line = part.trim();
                    if (!line.startsWith("data:")) continue;

                    let payload;
                    try {
                        payload = JSON.parse(line.slice("data:".length).trim());
                    } catch {
                        continue;
                    }

                    if (payload.error) {
                        settle(onError, payload.error);
                        return;
                    }

                    if (payload.done) {
                        settle(onDone, {
                            answer: payload.answer,
                            sources: payload.sources || [],
                        });
                        return;
                    }

                    if (payload.token !== undefined) {
                        onToken(payload.token);
                    }
                }
            }
        } catch {
            settle(onError, "The connection was lost. Please try again.");
            return;
        }

        settle(onError, "The answer stopped unexpectedly. Please try again.");
    },
};
