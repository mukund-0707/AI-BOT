const Upload = {
    allowedExtensions: [".pdf", ".docx", ".txt"],
    maxSizeBytes: 10 * 1024 * 1024,

    init() {
    },

    validateFile(file) {
        const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

        if (!this.allowedExtensions.includes(extension)) {
            return "Only PDF, DOCX and TXT files are allowed.";
        }

        if (file.size === 0) {
            return "Uploaded file is empty.";
        }

        if (file.size > this.maxSizeBytes) {
            return "Maximum file size is 10 MB.";
        }

        return null;
    },

    async handleFile(file) {
        UI.setUploading(true);
        UI.setDocumentStatus("uploading");

        Chat.clearMessages();
        Chat.renderStatusMessage(`Uploading "${file.name}"...`);

        try {
            const uploadResult = await Api.uploadDocument(file);

            UI.setDocumentStatus("processing", uploadResult.id, uploadResult.original_name);
            Chat.renderStatusMessage(`Processing "${uploadResult.original_name}"...`);

            const processResult = await Api.processDocument(uploadResult.id);

            if (processResult.status !== "ready") {
                throw new Error("Document processing failed. Please try again.");
            }

            UI.setDocumentStatus("ready", processResult.id, uploadResult.original_name);

            Chat.renderStatusMessage(
                `"${uploadResult.original_name}" is ready. Ask your first question!`,
                "success"
            );
        } catch (error) {
            UI.setDocumentStatus("failed");
            Chat.renderStatusMessage(error.message, "error");
        } finally {
            UI.setUploading(false);
        }
    },
};
