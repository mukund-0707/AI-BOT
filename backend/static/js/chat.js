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

        if (UI.elements.newChatButton) {
            UI.elements.newChatButton.addEventListener("click", () => {
                this.startNewConversation();
            });
        }
    },

    async startNewConversation() {
        // Mid-answer the reply would land in a conversation that no longer exists.
        if (AppState.isSending) {
            return;
        }

        try {
            await Api.resetConversation();
            this.renderWelcomeMessage();
        } catch (error) {
            this.renderStatusMessage(error.message, "error");
        }
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

        // Create the bot bubble early — tokens will be appended into it.
        let streamNode = null;
        let streamBody = null;
        let rawTokens = "";

        // Markdown.render re-parses the whole answer, so doing it per token is
        // quadratic and visibly janky on a long reply. One render per frame
        // looks identical and costs a fraction of the work.
        let pendingFrame = null;

        const paint = () => {
            pendingFrame = null;
            streamBody.innerHTML = Markdown.render(rawTokens);
            UI.scrollToBottom();
        };

        const cancelPendingPaint = () => {
            if (pendingFrame !== null) {
                cancelAnimationFrame(pendingFrame);
                pendingFrame = null;
            }
        };

        const onToken = (token) => {
            // Remove the thinking indicator on first token.
            this.removeThinkingMessage();

            if (!streamNode) {
                streamNode = this.appendNode(
                    "message ai-message",
                    `
                    ${this.botAvatarHtml(true)}
                    <div class="message-content">
                        <div class="message-body"></div>
                    </div>
                    `
                );
                streamBody = streamNode.querySelector(".message-body");
            }

            rawTokens += token;

            if (pendingFrame === null) {
                pendingFrame = requestAnimationFrame(paint);
            }
        };

        const onDone = ({ answer, sources }) => {
            // A queued frame still holds the partial text and would paint over
            // the final answer after it lands.
            cancelPendingPaint();

            // Replace the streamed content with the final authoritative answer
            // (handles the replace:true case where the answer was NO_ANSWER).
            if (streamBody) {
                streamBody.innerHTML = Markdown.render(answer);
            } else {
                // Stream never emitted tokens (e.g. small talk fell back).
                this.removeThinkingMessage();
                streamNode = this.appendNode(
                    "message ai-message",
                    `
                    ${this.botAvatarHtml(true)}
                    <div class="message-content">
                        <div class="message-body">${Markdown.render(answer)}</div>
                    </div>
                    `
                );
            }

            // Append sources block if present.
            if (sources && sources.length > 0) {
                const items = sources
                    .map((source) => {
                        const page =
                            source.page_number != null
                                ? `Page ${source.page_number}`
                                : "Page unknown";
                        return `<li>${this.escapeHtml(source.document_name)} · ${page}</li>`;
                    })
                    .join("");

                const sourcesDiv = document.createElement("div");
                sourcesDiv.className = "message-sources";
                sourcesDiv.innerHTML = `
                    <span class="message-sources-label">Sources</span>
                    <ul>${items}</ul>
                `;
                streamNode.querySelector(".message-content").appendChild(sourcesDiv);
            }

            UI.scrollToBottom();
            UI.setSending(false);
        };

        const onError = (message) => {
            cancelPendingPaint();
            this.removeThinkingMessage();
            if (streamNode) streamNode.remove();
            this.renderStatusMessage(message, "error");
            UI.setSending(false);
        };

        await Api.askQuestionStream(question, onToken, onDone, onError);
    },
};

document.addEventListener("DOMContentLoaded", () => {
    UI.init();
    Chat.init();
    Upload.init();
});
