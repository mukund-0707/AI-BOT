const Theme = {
    storageKey: "mukii-theme",

    init() {
        this.button = document.getElementById("themeButton");
        if (!this.button) {
            return;
        }

        const savedTheme = localStorage.getItem(this.storageKey) || "light";
        this.applyTheme(savedTheme);

        this.button.addEventListener("click", () => {
            const nextTheme =
                document.documentElement.dataset.theme === "dark"
                    ? "light"
                    : "dark";
            this.applyTheme(nextTheme);
            localStorage.setItem(this.storageKey, nextTheme);
        });
    },

    applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        this.button.textContent = theme === "dark" ? "☀️" : "🌙";
        this.button.title =
            theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    },
};

document.addEventListener("DOMContentLoaded", () => {
    Theme.init();
});
