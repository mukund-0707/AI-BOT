/**
 * Minimal, dependency-free Markdown renderer for chat answers.
 *
 * The model replies in Markdown (bold, numbered lists, headings), so the raw
 * text cannot be dropped into the DOM as-is. Everything is HTML-escaped first
 * and only the generated tags are trusted, so model output can never inject
 * markup.
 */
const Markdown = {
    // Placeholders for extracted code. Control characters survive escaping and
    // cannot appear in the model's own text, so they never collide with it.
    BLOCK_MARK: "\u0000",
    SPAN_MARK: "\u0001",

    // Placeholder for an escaped pipe inside a table cell.
    PIPE_MARK: "\u0002",

    /**
     * Cells of one table row. GFM makes the outer pipes optional, so they are
     * stripped when present rather than turned into empty cells.
     */
    splitRow(line) {
        const row = line
            .trim()
            .replace(/^\|/, "")
            .replace(/\|$/, "");

        return row
            .replace(/\\\|/g, this.PIPE_MARK)
            .split("|")
            .map((cell) => cell.trim().split(this.PIPE_MARK).join("|"));
    },

    isTableRow(line) {
        return line.includes("|") && this.splitRow(line).length >= 2;
    },

    /** The `| --- | :---: |` line under a header row. */
    isTableSeparator(line) {
        if (!line.includes("-")) {
            return false;
        }

        const cells = this.splitRow(line);

        return cells.length >= 2 && cells.every((cell) => /^:?-+:?$/.test(cell));
    },

    cellAlignment(cell) {
        const left = cell.startsWith(":");
        const right = cell.endsWith(":");

        if (left && right) {
            return "center";
        }

        return right ? "right" : left ? "left" : "";
    },

    /**
     * A table starts at a pipe row followed by a separator, or - because the
     * model does not always emit the separator - by another row of the same
     * width. Requiring equal width keeps a prose line that happens to contain a
     * pipe out of the table branch.
     */
    startsTable(line, nextLine) {
        if (!this.isTableRow(line)) {
            return false;
        }

        if (this.isTableSeparator(nextLine)) {
            return true;
        }

        return (
            this.isTableRow(nextLine) &&
            this.splitRow(nextLine).length === this.splitRow(line).length
        );
    },

    /** Renders the table at `start`, returning it with the last line it used. */
    consumeTable(lines, start) {
        const header = this.splitRow(lines[start]);

        let index = start;
        let alignments = [];

        const separator = (lines[start + 1] || "").trim();

        if (this.isTableSeparator(separator)) {
            alignments = this.splitRow(separator).map((cell) =>
                this.cellAlignment(cell)
            );
            index = start + 1;
        }

        const rows = [];

        while (index + 1 < lines.length) {
            const candidate = lines[index + 1].trim();

            if (!this.isTableRow(candidate) || this.isTableSeparator(candidate)) {
                break;
            }

            rows.push(this.splitRow(candidate));
            index += 1;
        }

        const cell = (tag, value, column) => {
            const align = alignments[column] || "";
            const style = align ? ` style="text-align:${align}"` : "";

            return `<${tag}${style}>${this.renderInline(value)}</${tag}>`;
        };

        const head = header
            .map((value, column) => cell("th", value, column))
            .join("");

        // Short rows are padded, so a ragged table still renders as a grid.
        const body = rows
            .map((row) => {
                const cells = header.map((ignored, column) =>
                    cell("td", row[column] === undefined ? "" : row[column], column)
                );

                return `<tr>${cells.join("")}</tr>`;
            })
            .join("");

        const html =
            '<div class="table-scroll"><table>' +
            `<thead><tr>${head}</tr></thead>` +
            (body ? `<tbody>${body}</tbody>` : "") +
            "</table></div>";

        return { html, index };
    },

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    render(text) {
        if (!text) {
            return "";
        }

        let source = String(text).replace(/\r\n/g, "\n");

        // Reasoning models sometimes leak their scratchpad into the answer.
        source = source.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

        const mark = this.BLOCK_MARK;
        const codeBlocks = [];

        source = source.replace(
            /```[^\n]*\n?([\s\S]*?)```/g,
            (match, code) => {
                codeBlocks.push(code.replace(/\n$/, ""));
                return `${mark}${codeBlocks.length - 1}${mark}`;
            }
        );

        const html = [];
        let paragraph = [];

        // Open lists, outermost first, so indented sub-lists can nest inside
        // the <li> that introduced them.
        const listStack = [];

        const closeParagraph = () => {
            if (paragraph.length > 0) {
                html.push(`<p>${this.renderInline(paragraph.join("\n"))}</p>`);
                paragraph = [];
            }
        };

        const closeLists = (indent = -1) => {
            while (
                listStack.length > 0 &&
                listStack[listStack.length - 1].indent > indent
            ) {
                html.push(`</li></${listStack.pop().type}>`);
            }
        };

        const addListItem = (indent, type, content) => {
            closeLists(indent);

            const current = listStack[listStack.length - 1];

            if (current && current.indent === indent) {
                if (current.type === type) {
                    html.push("</li>");
                } else {
                    html.push(`</li></${listStack.pop().type}>`);
                    html.push(`<${type}>`);
                    listStack.push({ type, indent });
                }
            } else {
                // First list, or one level deeper than the item above it.
                html.push(`<${type}>`);
                listStack.push({ type, indent });
            }

            html.push(`<li>${this.renderInline(content)}`);
        };

        const closeList = () => closeLists();

        const blockPattern = new RegExp(`^${mark}(\\d+)${mark}$`);

        // Indexed rather than for-of: a table spans several lines, so its branch
        // has to look ahead and then skip what it consumed.
        const lines = source.split("\n");

        for (let cursor = 0; cursor < lines.length; cursor += 1) {
            const expanded = lines[cursor].replace(/\t/g, "    ");
            const indent = expanded.length - expanded.trimStart().length;
            const line = expanded.trim();

            if (line === "") {
                closeParagraph();
                closeList();
                continue;
            }

            const codeBlock = line.match(blockPattern);
            if (codeBlock) {
                closeParagraph();
                closeList();
                const code = this.escapeHtml(codeBlocks[Number(codeBlock[1])]);
                html.push(`<pre><code>${code}</code></pre>`);
                continue;
            }

            const heading = line.match(/^(#{1,6})\s+(.*)$/);
            if (heading) {
                closeParagraph();
                closeList();
                // Answers live inside a chat bubble, so demote heading sizes.
                const level = Math.min(heading[1].length + 2, 6);
                html.push(
                    `<h${level}>${this.renderInline(heading[2])}</h${level}>`
                );
                continue;
            }

            if (/^([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
                closeParagraph();
                closeList();
                html.push("<hr>");
                continue;
            }

            const orderedItem = line.match(/^\d+[.)]\s+(.*)$/);
            if (orderedItem) {
                closeParagraph();
                addListItem(indent, "ol", orderedItem[1]);
                continue;
            }

            const bulletItem = line.match(/^[-*+•]\s+(.*)$/);
            if (bulletItem) {
                closeParagraph();
                addListItem(indent, "ul", bulletItem[1]);
                continue;
            }

            const quote = line.match(/^>\s?(.*)$/);
            if (quote) {
                closeParagraph();
                closeList();
                html.push(
                    `<blockquote>${this.renderInline(quote[1])}</blockquote>`
                );
                continue;
            }

            // Checked last: every other block type is decided by the first
            // character, while a table is only recognisable from two lines.
            if (this.startsTable(line, (lines[cursor + 1] || "").trim())) {
                closeParagraph();
                closeList();

                const table = this.consumeTable(lines, cursor);
                html.push(table.html);
                cursor = table.index;
                continue;
            }

            closeList();
            paragraph.push(line);
        }

        closeParagraph();
        closeList();

        return html.join("");
    },

    renderInline(text) {
        const mark = this.SPAN_MARK;
        let out = this.escapeHtml(text);

        const codeSpans = [];
        out = out.replace(/`([^`]+)`/g, (match, code) => {
            codeSpans.push(code);
            return `${mark}${codeSpans.length - 1}${mark}`;
        });

        out = out
            .replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>")
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
            .replace(/__([^_]+)__/g, "<strong>$1</strong>")
            .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
            .replace(
                /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
            )
            .replace(/\n/g, "<br>");

        return out.replace(
            new RegExp(`${mark}(\\d+)${mark}`, "g"),
            (match, index) => `<code>${codeSpans[Number(index)]}</code>`
        );
    },
};
