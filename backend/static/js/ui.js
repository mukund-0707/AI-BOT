const AppState = {
    documentId: null,
    documentName: null,
    documentStatus: null,
    isSending: false,
    isUploading: false,
};

const UI = {
    elements: {},

    init() {
        this.elements = {
            chatWrapper: document.getElementById("chatWrapper"),
            messageInput: document.getElementById("messageInput"),
            sendButton: document.getElementById("sendButton"),
            newChatButton: document.getElementById("newChatButton"),
        };

        this.elements.messageInput.addEventListener("input", () => {
            this.updateControls();
        });
    },

    setDocumentStatus(status, documentId = null, documentName = null) {
        AppState.documentStatus = status;

        if (documentId !== null) {
            AppState.documentId = documentId;
        }

        if (documentName !== null) {
            AppState.documentName = documentName;
        }

        this.updateControls();
    },

    setSending(isSending) {
        AppState.isSending = isSending;
        this.updateControls();
    },

    setUploading(isUploading) {
        AppState.isUploading = isUploading;
        this.updateControls();
    },

    canSend() {
        const question = this.elements.messageInput.value.trim();

        return (
            question.length > 0 &&
            !AppState.isSending
        );
    },

    updateControls() {
        const isBusy = AppState.isSending;

        this.elements.messageInput.disabled = isBusy;
        this.elements.sendButton.disabled = !this.canSend();

        if (AppState.isSending) {
            this.elements.messageInput.placeholder = "Mukii is thinking...";
        } else {
            this.elements.messageInput.placeholder = "Ask a question about the documents...";
        }
    },

    scrollToBottom() {
        const chatArea = document.querySelector(".chat-area");
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    },
};
