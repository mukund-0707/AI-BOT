const Chat = {
    botAvatarSrc: "",

    init() {
        this.botAvatarSrc =
            UI.elements.chatWrapper.dataset.botAvatar ||
            "/static/images/mukii-bot.svg";

        this.renderWelcomeMessage();

        UI.elements.sendButton.addEventListener("click", () => {
            this.sendMessage();
        });

        UI.elements.messageInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                this.sendMessage();
            }
        });
    },

    escapeHtml(text) {
        return Markdown.escapeHtml(text);
    },

    appendNode(className, innerHtml) {
        const node = document.createElement("div");
        node.className = className;
        node.innerHTML = innerHtml;

        UI.elements.chatWrapper.appendChild(node);
        UI.scrollToBottom();

        return node;
    },

    botAvatarHtml(small = false) {
        const className = small ? "bot-avatar small" : "bot-avatar";

        return `
            <div class="${className}">
                <img
                    src="${this.botAvatarSrc}"
                    alt="Mukii"
                    class="bot-avatar-img"
                >
            </div>
        `;
    },

    clearMessages() {
        UI.elements.chatWrapper.innerHTML = "";
    },

    renderWelcomeMessage() {
        UI.elements.chatWrapper.innerHTML = `
            <div class="welcome-message">
                ${this.botAvatarHtml()}
                <div class="welcome-content">
                    <h2>Hello! I'm Mukii</h2>
                    <p>I'm here to answer your questions.</p>
                </div>
            </div>
        `;
    },

    renderStatusMessage(message, type = "info") {
        const statusClass =
            type === "success"
                ? "status-message success"
                : type === "error"
                    ? "status-message error"
                    : "status-message";

        this.appendNode(
            statusClass,
            `
            ${this.botAvatarHtml(true)}
            <div class="status-content">${this.escapeHtml(message)}</div>
        `
        );
    },

    renderUserMessage(message) {
        this.appendNode(
            "message user-message",
            `<div class="message-content">${this.escapeHtml(message)}</div>`
        );
    },

    renderAIMessage(message, sources = []) {
        let sourcesHtml = "";

        if (sources.length > 0) {
            const items = sources
                .map((source) => {
                    const page =
                        source.page_number != null
                            ? `Page ${source.page_number}`
                            : "Page unknown";
                    return `<li>${this.escapeHtml(source.document_name)} · ${page}</li>`;
                })
                .join("");

            sourcesHtml = `
                <div class="message-sources">
                    <span class="message-sources-label">Sources</span>
                    <ul>${items}</ul>
                </div>
            `;
        }

        this.appendNode(
            "message ai-message",
            `
            ${this.botAvatarHtml(true)}
            <div class="message-content">
                <div class="message-body">${Markdown.render(message)}</div>
                ${sourcesHtml}
            </div>
        `
        );
    },

    renderThinkingMessage() {
        const thinkingNode = this.appendNode(
            "message ai-message thinking-message",
            `
            ${this.botAvatarHtml(true)}
            <div class="message-content thinking-content">
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
                <span class="thinking-dot"></span>
            </div>
        `
        );

        thinkingNode.id = "thinkingMessage";
    },

    removeThinkingMessage() {
        const thinkingNode = document.getElementById("thinkingMessage");
        if (thinkingNode) {
            thinkingNode.remove();
        }
    },

    async sendMessage() {
        if (!UI.canSend()) {
            return;
        }

        const question = UI.elements.messageInput.value.trim();
        UI.elements.messageInput.value = "";
        UI.updateControls();

        this.renderUserMessage(question);
        UI.setSending(true);
        this.renderThinkingMessage();

        try {
            const result = await Api.askQuestion(AppState.documentId, question);

            this.removeThinkingMessage();
            this.renderAIMessage(
                result.answer || "No answer received.",
                result.sources || []
            );
        } catch (error) {
            this.removeThinkingMessage();
            this.renderStatusMessage(error.message, "error");
        } finally {
            UI.setSending(false);
        }
    },
};

document.addEventListener("DOMContentLoaded", () => {
    UI.init();
    Chat.init();
    Upload.init();
});
