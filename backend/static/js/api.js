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

    askQuestion(documentId, question) {
        return this.request("/api/chat/ask/", {
            method: "POST",
            body: JSON.stringify({
                document_id: documentId,
                question,
            }),
        });
    },
};
