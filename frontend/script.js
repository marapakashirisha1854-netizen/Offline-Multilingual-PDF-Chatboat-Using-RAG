// ============================================================
// Offline Multilingual PDF Chatbot - script.js
// ============================================================

const API_BASE = "";

let uploadedDocuments = [];
let currentSpeech = null;
let currentAudio = null;
let currentSpeechButton = null;
let isSpeaking = false;
let mediaRecorder = null;
let audioChunks = [];
let microphoneStream = null;
let isRecording = false;
let activeView = "home";
let quizReturnView = "home";


// ============================================================
// DOM HELPERS
// ============================================================

const $ = (id) => document.getElementById(id);

function query(selector) {
    return document.querySelector(selector);
}

function queryAll(selector) {
    return document.querySelectorAll(selector);
}


// ============================================================
// INITIAL LOAD
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    loadDocuments();

    setupNavigation();

    setupLanguageControls();

    setupUpload();

    setupChat();

    setupVoiceInput();

    setupSidebar();

    setupKeyboard();

    setupSummarizeModal();

});


// ============================================================
// LOAD DOCUMENTS
// ============================================================

async function loadDocuments() {

    try {

        const response = await fetch(
            `${API_BASE}/api/documents`
        );

        if (!response.ok) {
            throw new Error("Unable to load documents.");
        }

        const data = await response.json();

        uploadedDocuments = data.documents || [];

        renderDocuments();

    } catch (error) {

        console.error(
            "Load documents error:",
            error
        );

    }
}


// ============================================================
// RENDER DOCUMENTS IN SIDEBAR
// ============================================================

function renderDocuments() {

    const list =
        $("uploadedPdfList");

    if (!list) {
        return;
    }

    list.innerHTML = "";

    const count =
        $("pdfCount");

    if (count) {
        count.textContent =
            uploadedDocuments.length;
    }

    if (uploadedDocuments.length === 0) {

        list.innerHTML = `
            <div class="empty-pdf-message">
                No PDFs uploaded yet
            </div>
        `;

        return;
    }

    uploadedDocuments.forEach(
        (filename) => {

            const item =
                document.createElement("div");

            item.className =
                "uploaded-pdf-item";

            item.innerHTML = `
                <div class="uploaded-pdf-info">

                    <div class="uploaded-pdf-icon">
                        <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        >
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <path d="M9 13h6"></path>
                            <path d="M9 17h6"></path>
                        </svg>
                    </div>

                    <div
                        class="uploaded-pdf-name"
                        title="${escapeHtml(filename)}"
                    >
                        ${escapeHtml(filename)}
                    </div>

                </div>

                <button
                    class="remove-pdf-btn"
                    type="button"
                    title="Remove PDF"
                    data-filename="${escapeHtml(filename)}"
                >
                    ×
                </button>
            `;

            const removeButton =
                item.querySelector(
                    ".remove-pdf-btn"
                );

            removeButton.addEventListener(
                "click",
                () => removeDocument(filename)
            );

            list.appendChild(item);
        }
    );
    updateQuizDocSelector();
}


// ============================================================
// UPLOAD SETUP
// ============================================================

function setupUpload() {

    const fileInput =
        $("pdfInput");

    const chooseButton =
        $("choosePdfBtn");

    if (!fileInput) {
        return;
    }

    if (chooseButton) {

        chooseButton.addEventListener(
            "click",
            () => fileInput.click()
        );

    }

    fileInput.addEventListener(
        "change",
        async (event) => {

            const files =
                Array.from(
                    event.target.files || []
                );

            if (!files.length) {
                return;
            }

            await uploadPDFs(files);

            fileInput.value = "";

        }
    );
}


// ============================================================
// UPLOAD PDFS
// ============================================================

async function uploadPDFs(files) {

    const formData =
        new FormData();

    files.forEach(
        (file) => {

            if (
                file.type === "application/pdf" ||
                file.name.toLowerCase().endsWith(".pdf")
            ) {

                formData.append(
                    "files",
                    file
                );

            }

        }
    );

    const status =
        $("uploadStatus");

    if (status) {

        status.textContent =
            "Uploading PDF...";

    }

    try {

        const response =
            await fetch(
                `${API_BASE}/api/upload`,
                {
                    method: "POST",
                    body: formData
                }
            );

        if (!response.ok) {

            throw new Error(
                "PDF upload failed."
            );

        }

        const data =
            await response.json();

        let successCount = 0;

        let failedCount = 0;
        const failures = [];

        (data.results || []).forEach(
            (result) => {

                if (result.success) {
                    successCount++;
                } else {
                    failedCount++;
                    failures.push(`${result.filename || "PDF"}: ${result.error || "processing failed"}`);
                }

            }
        );

        await loadDocuments();

        if (status) {

            if (failedCount === 0) {

                status.textContent =
                    `${successCount} PDF uploaded successfully.`;

            } else {

                status.textContent =
                    `${successCount} uploaded, ${failedCount} failed. ${failures.join(" ")}`;

            }

        }

        showToast(failedCount ? failures[0] : "PDF uploaded successfully.", failedCount ? "error" : "success");

    } catch (error) {

        console.error(
            "Upload error:",
            error
        );

        if (status) {

            status.textContent =
                "PDF upload failed.";

        }

        showToast(
            error.message,
            "error"
        );

    }
}


// ============================================================
// REMOVE SINGLE DOCUMENT
// ============================================================

async function removeDocument(
    filename
) {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/documents/${encodeURIComponent(filename)}`,
                {
                    method: "DELETE"
                }
            );

        if (!response.ok) {

            throw new Error(
                "Unable to remove PDF."
            );

        }

        await loadDocuments();

        // Do NOT add "removed successfully"
        // message to chat.

        showToast(
            "PDF removed.",
            "success"
        );

    } catch (error) {

        console.error(
            "Remove PDF error:",
            error
        );

        showToast(
            error.message,
            "error"
        );

    }
}


// ============================================================
// CLEAR ALL DOCUMENTS
// ============================================================

async function clearAllDocuments() {

    if (
        uploadedDocuments.length === 0
    ) {

        showToast(
            "No PDFs to clear.",
            "info"
        );

        return;
    }

    const confirmed =
        window.confirm(
            "Remove all uploaded PDFs?"
        );

    if (!confirmed) {
        return;
    }

    try {

        const response =
            await fetch(
                `${API_BASE}/api/documents`,
                {
                    method: "DELETE"
                }
            );

        if (!response.ok) {

            throw new Error(
                "Unable to clear PDFs."
            );

        }

        uploadedDocuments = [];

        renderDocuments();

        showToast(
            "All PDFs cleared.",
            "success"
        );

    } catch (error) {

        console.error(
            "Clear PDFs error:",
            error
        );

        showToast(
            error.message,
            "error"
        );

    }
}


// ============================================================
// NAVIGATION
// ============================================================

function setupNavigation() {

    const navItems =
        queryAll(".nav-item");

    navItems.forEach(
        (item) => {

            item.addEventListener(
                "click",
                () => {

                    const page =
                        item.dataset.page;

                    if (!page) {
                        return;
                    }

                    navigateTo(page);

                }
            );

        }
    );
}


function navigateTo(page) {

    const views =
        queryAll(".home-view, .chat-view, .special-view");

    views.forEach(
        (view) => {

            view.classList.add(
                "hidden"
            );

        }
    );

    const target =
        $(`${page}View`);

    if (target) {

        target.classList.remove(
            "hidden"
        );
        activeView = page;

    }

    const chatInput = $("chatInputArea");
    if (chatInput) {
        if (page === "quiz" || page === "settings" || page === "help") {
            chatInput.classList.add("hidden");
        } else {
            chatInput.classList.remove("hidden");
        }
    }

    const navItems =
        queryAll(".nav-item");

    navItems.forEach(
        (item) => {

            item.classList.remove(
                "active"
            );

            if (
                item.dataset.page === page
            ) {

                item.classList.add(
                    "active"
                );

            }

        }
    );

    closeMobileSidebar();
}

function resetConversation() {
    const messages = $("chatMessages");
    const input = $("questionInput");
    if (messages) messages.innerHTML = "";
    if (input) input.value = "";
    const status = $("uploadStatus");
    if (status) status.textContent = "";
}

function goHome() {
    resetConversation();
    navigateTo("home");
}

function newChat() {
    resetConversation();
    navigateTo("home");
}

function showSettings() { navigateTo("settings"); }
function showHelp() { navigateTo("help"); }
function syncSettingsLanguage(value) {
    changeLanguage(value);
}
function updateQuizDocSelector() {
    const select = $("quizDocSelect");
    if (!select) return;
    const currentVal = select.value;
    const docs = uploadedDocuments || [];

    let html = `<option value="__all__">All Uploaded PDFs</option>`;
    docs.forEach(doc => {
        html += `<option value="${escapeHtml(doc)}">${escapeHtml(doc)}</option>`;
    });
    select.innerHTML = html;

    if (currentVal && (currentVal === "__all__" || docs.includes(currentVal))) {
        select.value = currentVal;
    } else if (docs.length === 1) {
        select.value = docs[0];
    } else {
        select.value = "__all__";
    }
}

function onQuizDocChange() {
    generateQuiz(true);
}

function onGenerateQuizClick() {
    generateQuiz(true);
}

function showQuiz() {
    quizReturnView = activeView === "quiz"
        ? quizReturnView
        : activeView;
    navigateTo("quiz");
    updateQuizDocSelector();
    const container = $("quizContent");
    if (!container || !container.querySelector(".quiz-question")) {
        generateQuiz();
    }
}
function backFromQuiz() {
    navigateTo(quizReturnView && quizReturnView !== "quiz" ? quizReturnView : "chat");
}
function clearAllPDFs() { return clearAllDocuments(); }
function showUploadedPDFs() {
    const panel = $("uploadedPdfList");
    if (panel) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function generateQuiz(force = false) {
    const container = $("quizContent");
    if (!container) return;
    if (!force && container.querySelector(".quiz-question")) {
        return;
    }
    const langVal = ($("answerLanguage") || {}).value || "English";
    const docSelect = $("quizDocSelect");
    const selectedDoc = docSelect ? docSelect.value : "__all__";

    const loadingTexts = {
        English: selectedDoc && selectedDoc !== "__all__"
            ? `Creating questions from ${selectedDoc}…`
            : "Creating questions from your uploaded PDFs…",
        Telugu: selectedDoc && selectedDoc !== "__all__"
            ? `${selectedDoc} నుండి ప్రశ్నలు రూపొందిస్తున్నాము…`
            : "మీరు అప్‌లోడ్ చేసిన PDFల నుండి ప్రశ్నలు రూపొందిస్తున్నాము…",
        Hindi: selectedDoc && selectedDoc !== "__all__"
            ? `${selectedDoc} से प्रश्न बनाए जा रहे हैं…`
            : "आपके अपलोड किए गए PDF से प्रश्न बनाए जा रहे हैं…"
    };
    const emptyTexts = {
        English: "Upload a PDF first, then open Quiz.",
        Telugu: "ముందుగా PDFని అప్‌లోడ్ చేయండి, ఆపై క్విజ్ తెరవండి.",
        Hindi: "पहले एक PDF अपलोड करें, फिर क्विज़ खोलें।"
    };
    if (!uploadedDocuments.length) {
        container.innerHTML = `<p class="empty-pdf-message">${escapeHtml(emptyTexts[langVal] || emptyTexts.English)}</p>`;
        return;
    }
    container.innerHTML = `<p>${escapeHtml(loadingTexts[langVal] || loadingTexts.English)}</p>`;
    try {
        const payload = {
            language: langVal,
            document: selectedDoc === "__all__" ? null : selectedDoc
        };
        const response = await fetch(`${API_BASE}/api/quiz`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Quiz generation failed.");
        const questionsMarkup = (data.questions || []).map((q, i) => `
            <fieldset class="quiz-question"><legend>${i + 1}. ${escapeHtml(q.question)}</legend>
            ${(q.options || []).map((option, j) => `<label><input type="radio" name="quiz-${escapeHtml(q.id || i)}" value="${j}"> ${String.fromCharCode(65+j)}) ${escapeHtml(option)}</label>`).join("")}
            <div class="quiz-feedback" data-answer="${Number(q.answer)}" hidden></div></fieldset>`).join("");
        const messages = {
            English: { correct: "Correct", wrong: "Wrong", your: "Your answer", answer: "Correct answer", none: "No answer selected", score: "Score", checkBtn: "Check answers", newBtn: "New Quiz" },
            Telugu: { correct: "సరైన సమాధానం", wrong: "తప్పు సమాధానం", your: "మీ సమాధానం", answer: "సరైన సమాధానం", none: "సమాధానం ఎంచుకోలేదు", score: "స్కోరు", checkBtn: "సమాధానాలను తనిఖీ చేయండి", newBtn: "కొత్త క్విజ్" },
            Hindi: { correct: "सही उत्तर", wrong: "गलत उत्तर", your: "आपका उत्तर", answer: "सही उत्तर", none: "कोई उत्तर नहीं चुना", score: "स्कोर", checkBtn: "उत्तर जाँचें", newBtn: "नया क्विज़" }
        };
        const labels = messages[langVal] || messages.English;
        container.innerHTML = `<div class="quiz-question-list">${questionsMarkup}</div>
            <div class="quiz-controls">
                <button type="button" class="choose-pdf-btn" id="gradeQuizBtn">${escapeHtml(labels.checkBtn)}</button>
                <button type="button" class="choose-pdf-btn" id="newQuizBtn" style="margin-left: 8px;">${escapeHtml(labels.newBtn)}</button>
                <p id="quizScore" aria-live="polite"></p>
            </div>`;
        const gradeButton = $("gradeQuizBtn");
        const newQuizButton = $("newQuizBtn");
        const scoreDisplay = $("quizScore");

        if (newQuizButton) {
            newQuizButton.addEventListener("click", () => {
                generateQuiz(true);
            });
        }

        gradeButton.addEventListener("click", () => {
            let score = 0;
            container.querySelectorAll(".quiz-question").forEach((fieldset, i) => {
                const feedback = fieldset.querySelector(".quiz-feedback");
                const correct = Number(feedback.dataset.answer);
                const qid = data.questions[i].id || i;
                const selected = fieldset.querySelector(`input[name="quiz-${qid}"]:checked`);
                feedback.hidden = false;
                const rightOption = `${String.fromCharCode(65 + correct)}) ${data.questions[i].options[correct]}`;
                const chosen = selected ? `${String.fromCharCode(65 + Number(selected.value))}) ${data.questions[i].options[Number(selected.value)]}` : labels.none;
                const isCorrect = !!selected && Number(selected.value) === correct;
                if (isCorrect) score++;
                feedback.className = `quiz-feedback ${isCorrect ? "correct" : "incorrect"}`;
                feedback.textContent = isCorrect
                    ? `${labels.correct}\n${labels.your}: ${chosen}`
                    : `${labels.wrong}\n${labels.your}: ${chosen}\n${labels.answer}: ${rightOption}`;
            });
            scoreDisplay.textContent = `${labels.score}: ${score}/${data.questions.length} (${Math.round(score * 100 / data.questions.length)}%)`;
            container.dataset.graded = "1";
            gradeButton.disabled = true;
        });

        container.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener("change", () => {
                if (container.dataset.graded === "1") {
                    gradeButton.disabled = false;
                }
            });
        });
    } catch (error) {
        const retryLabels = {
            English: "Retry Quiz",
            Telugu: "మళ్ళీ ప్రయత్నించండి",
            Hindi: "पुनः प्रयास करें"
        };
        const retryLabel = retryLabels[langVal] || retryLabels.English;
        container.innerHTML = `
            <p class="answer-paragraph">${escapeHtml(error.message)}</p>
            <div style="margin-top: 14px;">
                <button type="button" class="choose-pdf-btn" id="retryQuizBtn">${escapeHtml(retryLabel)}</button>
            </div>
        `;
        $("retryQuizBtn")?.addEventListener("click", () => generateQuiz(true));
    }
}


// ============================================================
// NEW CHAT
// ============================================================

function startNewChat() {

    const messages =
        $("chatMessages");

    if (messages) {

        messages.innerHTML = "";

    }

    navigateTo("chat");

}


// ============================================================
// CHAT SETUP
// ============================================================

function setupChat() {

    const sendButton =
        $("sendBtn");

    const questionInput =
        $("questionInput");

    if (sendButton) {

        sendButton.addEventListener(
            "click",
            sendQuestion
        );

    }

    if (questionInput) {

        questionInput.addEventListener(
            "keydown",
            (event) => {

                if (
                    event.key === "Enter" &&
                    !event.shiftKey
                ) {

                    event.preventDefault();

                    sendQuestion();

                }

            }
        );

    }
}


// ============================================================
// SEND QUESTION
// ============================================================

async function sendQuestion() {

    const input =
        $("questionInput");

    if (!input) {
        return;
    }

    const question =
        input.value.trim();

    if (!question) {
        return;
    }

    if (
        uploadedDocuments.length === 0
    ) {

        showToast(
            "Please upload a PDF first.",
            "info"
        );

        return;
    }

    const languageSelect =
        $("answerLanguage");

    const language =
        languageSelect
            ? languageSelect.value
            : "English";

    input.value = "";

    navigateTo("chat");

    // Add user message immediately
    addUserMessage(question);

    // Create empty bot bubble immediately
    // No "Processing..." text.
    const botMessage =
        createBotMessage();

    const answerContent =
        botMessage.querySelector(
            ".answer-content"
        );

    const sourcesContainer =
        botMessage.querySelector(
            ".sources-container"
        );

    try {

        const response =
            await fetch(
                `${API_BASE}/api/ask`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        {
                            question: question,
                            language: language
                        }
                    )
                }
            );

        if (!response.ok) {

            let errorText =
                "Unable to get answer.";

            try {

                const errorData =
                    await response.json();

                errorText =
                    errorData.detail ||
                    errorText;

            } catch (_) {}

            throw new Error(
                errorText
            );
        }

        if (!response.body) {

            throw new Error(
                "No response received."
            );

        }

        await readNDJSONStream(
            response,
            answerContent,
            sourcesContainer
        );

    } catch (error) {

        console.error(
            "Question error:",
            error
        );

        answerContent.innerHTML = `
            <div class="answer-paragraph">
                ${escapeHtml(
                    error.message ||
                    "Unable to get answer."
                )}
            </div>
        `;

    }

    scrollChatToBottom();
}


// ============================================================
// ADD USER MESSAGE
// ============================================================

function addUserMessage(
    question
) {

    const messages =
        $("chatMessages");

    if (!messages) {
        return;
    }

    const row =
        document.createElement("div");

    row.className =
        "message-row user-row";

    row.innerHTML = `
        <div class="message user-message">
            <div class="question-content">
                ${escapeHtml(question)}
            </div>
            <div class="message-tools">
                <button type="button" class="message-tool" data-action="edit" aria-label="Edit question" title="Edit question"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z"/></svg></button>
                <button type="button" class="message-tool" data-action="copy" aria-label="Copy question" title="Copy question"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg></button>
            </div>
        </div>
    `;

    row.querySelector('[data-action="edit"]').addEventListener("click", () => {
        const input = $("questionInput");
        if (!input) return;
        input.value = question;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
        const next = row.nextElementSibling;
        row.remove();
        if (next?.classList.contains("bot-row")) next.remove();
    });
    row.querySelector('[data-action="copy"]').addEventListener("click", () => copyText(question));

    messages.appendChild(row);

    scrollChatToBottom();
}


// ============================================================
// CREATE BOT MESSAGE
// ============================================================

function createBotMessage() {

    const messages =
        $("chatMessages");

    const row =
        document.createElement("div");

    row.className =
        "message-row bot-row";

    row.innerHTML = `
        <div class="message bot-message">

            <div class="answer-content"><div class="typing-indicator" aria-label="Thinking"><span></span><span></span><span></span></div></div>

            <div class="message-tools hidden"><button type="button" class="message-tool" data-action="copy" aria-label="Copy answer" title="Copy answer"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg></button></div>

            <div class="sources-container hidden">
                <div class="sources-title">
                    Sources
                </div>

                <div class="sources-list"></div>
            </div>

            <div class="answer-actions">
                <button
                    type="button"
                    class="listen-btn"
                >
                    🔊 Listen
                </button>
            </div>

        </div>
    `;

    messages.appendChild(row);

    const tools = row.querySelector(".message-tools");
    if (tools) tools.hidden = true;

    row.querySelector('[data-action="copy"]').addEventListener("click", () => {
        const answer = row.querySelector(".answer-content")?.innerText || "";
        copyText(answer);
    });

    const listenButton =
        row.querySelector(
            ".listen-btn"
        );

    listenButton.addEventListener(
        "click",
        () => {

            const answer =
                row.querySelector(
                    ".answer-content"
                ).innerText.trim();

            if (!answer) {
                return;
            }

            toggleSpeech(
                answer,
                listenButton
            );

        }
    );
    listenButton.hidden = true;

    return row.querySelector(
        ".bot-message"
    );
}


// ============================================================
// READ NDJSON STREAM
// ============================================================

async function readNDJSONStream(
    response,
    answerElement,
    sourcesContainer
) {

    const reader =
        response.body.getReader();

    const decoder =
        new TextDecoder(
            "utf-8"
        );

    let buffer = "";

    let rawAnswer = "";

    while (true) {

        const {
            value,
            done
        } = await reader.read();

        if (done) {
            break;
        }

        buffer += decoder.decode(
            value,
            {
                stream: true
            }
        );

        const lines =
            buffer.split("\n");

        buffer =
            lines.pop() || "";

        for (const line of lines) {

            if (!line.trim()) {
                continue;
            }

            try {

                const data =
                    JSON.parse(line);

                if (
                    data.type === "text"
                ) {

                    rawAnswer +=
                        data.content || "";

                    answerElement.innerHTML =
                        formatAnswer(
                            rawAnswer
                        );
                    if (rawAnswer.trim()) {
                        const botMsg = answerElement.closest(".bot-message");
                        const listenButton = botMsg?.querySelector(".listen-btn");
                        if (listenButton) listenButton.hidden = false;
                        const tools = botMsg?.querySelector(".message-tools");
                        if (tools) {
                            tools.hidden = false;
                            tools.classList.remove("hidden");
                        }
                    }

                    scrollChatToBottom();

                } else if (
                    data.type === "sources"
                ) {

                    renderSources(
                        data.sources || [],
                        sourcesContainer
                    );

                }

            } catch (error) {

                console.warn(
                    "Invalid stream line:",
                    line
                );

            }

        }

    }

    // Handle final buffered line
    if (buffer.trim()) {

        try {

            const data =
                JSON.parse(
                    buffer
                );

            if (
                data.type === "text"
            ) {

                rawAnswer +=
                    data.content || "";

                answerElement.innerHTML =
                    formatAnswer(
                        rawAnswer
                    );
                if (rawAnswer.trim()) {
                    const botMsg = answerElement.closest(".bot-message");
                    const listenButton = botMsg?.querySelector(".listen-btn");
                    if (listenButton) listenButton.hidden = false;
                    const tools = botMsg?.querySelector(".message-tools");
                    if (tools) {
                        tools.hidden = false;
                        tools.classList.remove("hidden");
                    }
                }

            }

            if (
                data.type === "sources"
            ) {

                renderSources(
                    data.sources || [],
                    sourcesContainer
                );

            }

        } catch (_) {}

    }

    // Final formatting
    if (rawAnswer.trim()) {
        answerElement.innerHTML =
            formatAnswer(
                rawAnswer
            );
        const botMsg = answerElement.closest(".bot-message");
        const listenButton = botMsg?.querySelector(".listen-btn");
        if (listenButton) listenButton.hidden = false;
        const tools = botMsg?.querySelector(".message-tools");
        if (tools) {
            tools.hidden = false;
            tools.classList.remove("hidden");
        }
    } else {
        answerElement.innerHTML = `
            <div class="answer-paragraph">
                I could not find this information in the uploaded documents.
            </div>
        `;
    }

    scrollChatToBottom();
}


// ============================================================
// FORMAT ANSWER
// ============================================================

function formatAnswer(
    text
) {

    if (!text) {
        return "";
    }

    let safe =
        escapeHtml(text);

    safe = safe.replace(
        /\r\n/g,
        "\n"
    );

    safe = safe.replace(
        /\r/g,
        "\n"
    );

    // Bold markdown
    safe = safe.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );

    // Headings
    const headingNames = [
        "Definition",
        "Characteristics",
        "Characteristic",
        "Advantages",
        "Disadvantages",
        "Features",
        "Properties",
        "Applications",
        "Application",
        "Important Key Points",
        "Important Definitions",
        "Important Concepts",
        "Overall Summary",
        "Summary"
    ];

    headingNames.forEach(
        (heading) => {

            const pattern =
                new RegExp(
                    `(^|\\n)${heading}\\s*:?\\s*(?=\\n|$)`,
                    "gi"
                );

            safe = safe.replace(
                pattern,
                `$1<span class="answer-heading">${heading}</span>`
            );

        }
    );

    safe = renderMarkdownTables(safe);

    // Numbered list
    safe = safe.replace(
        /(?:^|\n)(\d+)\.\s+(.+?)(?=\n|$)/g,
        `<div class="answer-list-item">
            <strong>$1.</strong> $2
        </div>`
    );

    // Bullet list
    safe = safe.replace(
        /(?:^|\n)[•*-]\s+(.+?)(?=\n|$)/g,
        `<div class="answer-list-item">
            • $1
        </div>`
    );

    // Remaining new lines
    safe = safe.replace(
        /\n\n+/g,
        "<br><br>"
    );

    safe = safe.replace(
        /\n/g,
        "<br>"
    );

    return safe;
}

function renderMarkdownTables(text) {
    const lines = text.split("\n");
    const output = [];
    for (let index = 0; index < lines.length; index++) {
        const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
        const isSeparator = (line) => /^\s*\|[\s:|\-]+\|\s*$/.test(line);
        if (!isTableRow(lines[index]) || index + 1 >= lines.length || !isSeparator(lines[index + 1])) {
            output.push(lines[index]);
            continue;
        }
        const rows = [lines[index]];
        index += 2;
        while (index < lines.length && isTableRow(lines[index])) rows.push(lines[index++]);
        index--;
        const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
        const header = cells(rows[0]);
        const body = rows.slice(1).map(cells);
        output.push(`<div class="answer-table-wrap"><table class="answer-table"><thead><tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${header.map((_, cellIndex) => `<td>${row[cellIndex] || ""}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    }
    return output.join("\n");
}


// ============================================================
// RENDER SOURCES
// ============================================================

function renderSources(
    sources,
    container
) {

    if (!container) {
        return;
    }

    const list =
        container.querySelector(
            ".sources-list"
        );

    if (!list) {
        return;
    }

    list.innerHTML = "";

    if (!sources.length) {

        container.classList.add(
            "hidden"
        );

        return;
    }

    sources.forEach(
        (source) => {

            const filename =
                source.filename ||
                source.source ||
                "Unknown PDF";

            const page =
                source.page ??
                "Unknown";

            const item =
                document.createElement("div");

            item.className =
                "source-item";

            item.innerHTML = `
                <div
                    class="source-file"
                    title="${escapeHtml(filename)}"
                >
                    ${escapeHtml(filename)}
                </div>

                <div class="source-page">
                    Page ${escapeHtml(
                        String(page)
                    )}
                </div>
            `;

            list.appendChild(item);

        }
    );

    container.classList.remove(
        "hidden"
    );
}


// ============================================================
// SUMMARY (MODAL SELECTION & EXECUTION)
// ============================================================

function setupSummarizeModal() {
    const modal = $("summarizeModal");
    const selectAllBtn = $("selectAllPdfsBtn");
    const deselectAllBtn = $("deselectAllPdfsBtn");
    const closeBtn = $("closeSummarizeModalBtn");
    const cancelBtn = $("cancelSummarizeBtn");
    const confirmBtn = $("confirmSummarizeBtn");
    const validationError = $("summarizeValidationError");

    if (selectAllBtn) {
        selectAllBtn.addEventListener("click", () => {
            modal?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
            if (validationError) validationError.classList.add("hidden");
        });
    }

    if (deselectAllBtn) {
        deselectAllBtn.addEventListener("click", () => {
            modal?.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
        });
    }

    if (closeBtn) closeBtn.addEventListener("click", closeSummarizeModal);
    if (cancelBtn) cancelBtn.addEventListener("click", closeSummarizeModal);

    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeSummarizeModal();
        });
    }

    if (confirmBtn) {
        confirmBtn.addEventListener("click", () => {
            const checkboxes = Array.from(modal.querySelectorAll('input[type="checkbox"]:checked'));
            const selected = checkboxes.map(cb => cb.value);
            if (selected.length === 0) {
                if (validationError) validationError.classList.remove("hidden");
                return;
            }
            closeSummarizeModal();
            executeSummarizePDF(selected);
        });
    }
}

function openSummarizeModal() {
    if (uploadedDocuments.length === 0) {
        showToast("Please upload a PDF first.", "info");
        return;
    }

    const modal = $("summarizeModal");
    const list = $("summarizePdfList");
    const validationError = $("summarizeValidationError");
    if (!modal || !list) return;

    if (validationError) validationError.classList.add("hidden");

    list.innerHTML = uploadedDocuments.map((doc, idx) => `
        <label class="modal-pdf-item">
            <input type="checkbox" name="summaryPdf" value="${escapeHtml(doc)}" checked id="sumPdf_${idx}">
            <span>${escapeHtml(doc)}</span>
        </label>
    `).join("");

    list.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener("change", () => {
            const anyChecked = !!list.querySelector('input[type="checkbox"]:checked');
            if (anyChecked && validationError) {
                validationError.classList.add("hidden");
            }
        });
    });

    modal.classList.remove("hidden");
}

function closeSummarizeModal() {
    const modal = $("summarizeModal");
    if (modal) modal.classList.add("hidden");
    const validationError = $("summarizeValidationError");
    if (validationError) validationError.classList.add("hidden");
}

function summarizePDF() {
    openSummarizeModal();
}

async function executeSummarizePDF(selectedDocs) {
    if (!selectedDocs || selectedDocs.length === 0) {
        showToast("Please select at least one PDF.", "warning");
        return;
    }

    const previousView = activeView;
    const previousConversationEntry = $("chatMessages")?.lastElementChild || null;
    navigateTo("chat");

    const docLabels = selectedDocs.join(", ");
    addUserMessage(`Summarize: ${docLabels}`);

    const botMessage = createBotMessage();

    const answerActions = botMessage.querySelector(".answer-actions");
    if (answerActions) {
        const backButton = document.createElement("button");
        backButton.type = "button";
        backButton.className = "quiz-back-btn summary-back-btn";
        backButton.setAttribute("aria-label", "Back to previous chat");
        backButton.textContent = "Back";
        backButton.addEventListener("click", () => {
            if (previousConversationEntry?.isConnected) {
                previousConversationEntry.scrollIntoView({ behavior: "smooth", block: "center" });
            } else if (previousView !== "chat") {
                navigateTo(previousView);
            }
        });
        answerActions.prepend(backButton);
    }

    const answerContent = botMessage.querySelector(".answer-content");
    const sourcesContainer = botMessage.querySelector(".sources-container");
    const languageSelect = $("answerLanguage");
    const language = languageSelect ? languageSelect.value : "English";

    try {
        const response = await fetch(`${API_BASE}/api/summarize`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                language: language,
                documents: selectedDocs,
                filenames: selectedDocs
            })
        });

        if (!response.ok) {
            let errorText = "Unable to summarize PDF.";
            try {
                const errorData = await response.json();
                errorText = errorData.detail || errorText;
            } catch (_) {}
            throw new Error(errorText);
        }

        await readNDJSONStream(
            response,
            answerContent,
            sourcesContainer
        );
    } catch (error) {
        answerContent.innerHTML = `
            <div class="answer-paragraph">
                ${escapeHtml(error.message)}
            </div>
        `;
    }
}


// ============================================================
// LANGUAGE CONTROLS
// ============================================================

function changeLanguage(value) {

    const allowed = ["English", "Telugu", "Hindi"];

    const selected = allowed.includes(value)
        ? value
        : "English";

    const answerLanguage = $("answerLanguage");
    const settingsLanguage = $("settingsLanguage");

    if (answerLanguage && answerLanguage.value !== selected) {
        answerLanguage.value = selected;
    }

    if (settingsLanguage && settingsLanguage.value !== selected) {
        settingsLanguage.value = selected;
    }

    syncVoiceLanguage();

    return selected;
}

function setupLanguageControls() {

    const answerLanguage =
        $("answerLanguage");

    const voiceLanguage =
        $("voiceLanguage");

    if (answerLanguage) {

        answerLanguage.addEventListener(
            "change",
            () => {

                changeLanguage(answerLanguage.value);

            }
        );

    }

    if (voiceLanguage) {

        voiceLanguage.addEventListener(
            "change",
            () => {
                updateVoiceLanguage();
            }
        );

    }

    syncVoiceLanguage();
}


function syncVoiceLanguage() {

    const answerLanguage =
        $("answerLanguage");

    const voiceLanguage =
        $("voiceLanguage");

    if (
        !answerLanguage ||
        !voiceLanguage
    ) {
        return;
    }

    if (
        voiceLanguage.value === "auto"
    ) {
        return;
    }

    // Do not force the user's voice selection.
}


function updateVoiceLanguage() {

    // Recognition language is selected
    // when microphone starts.

}


// ============================================================
// VOICE INPUT
// ============================================================

function setupVoiceInput() {

    const voiceButton =
        $("voiceBtn");

    if (!voiceButton) {
        return;
    }

    voiceButton.addEventListener(
        "click",
        toggleVoiceRecognition
    );
}


function getRecognitionLanguage() {

    const voiceLanguage =
        $("voiceLanguage");

    const answerLanguage =
        $("answerLanguage");

    const selected =
        voiceLanguage
            ? voiceLanguage.value
            : "auto";

    if (selected === "en-IN") return "en";
    if (selected === "te-IN") return "te";
    if (selected === "hi-IN") return "hi";
    return "auto";
}


function toggleVoiceRecognition() {

    if (isRecording) {

        stopVoiceRecognition();

    } else {

        startVoiceRecognition();

    }
}


function startVoiceRecognition() {
    const voiceButton =
        $("voiceBtn");
    if (isRecording) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        showToast("Local microphone recording is not supported in this browser.", "error");
        return;
    }
    navigator.mediaDevices.getUserMedia({
        audio: {
            autoGainControl: { ideal: true },
            noiseSuppression: { ideal: true },
            echoCancellation: { ideal: true },
            channelCount: { ideal: 1 }
        }
    }).then((stream) => {
        microphoneStream = stream;
        audioChunks = [];
        const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
            .find((type) => MediaRecorder.isTypeSupported(type));
        mediaRecorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);
        mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data && event.data.size) audioChunks.push(event.data);
        });
        mediaRecorder.addEventListener("stop", async () => {
            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
            stream.getTracks().forEach((track) => track.stop());
            microphoneStream = null;
            isRecording = false;
            voiceButton.classList.remove("recording");
            voiceButton.setAttribute("aria-label", "Voice input");
            if (blob.size) await transcribeAudio(blob);
            else showToast("No audio was recorded. Please try again.", "error");
        });
        mediaRecorder.addEventListener("error", () => {
            stream.getTracks().forEach((track) => track.stop());
            microphoneStream = null;
            isRecording = false;
            voiceButton.classList.remove("recording");
            showToast("Microphone recording failed.", "error");
        });
        mediaRecorder.start(250);
        isRecording = true;
        voiceButton.classList.add("recording");
        voiceButton.setAttribute("aria-label", "Stop voice input");
        showToast("Recording locally. Click the microphone again to transcribe.", "info");
    }).catch((error) => {
        console.error("Microphone permission error:", error);
        if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
        microphoneStream = null;
        isRecording = false;
        voiceButton?.classList.remove("recording");
        voiceButton?.setAttribute("aria-label", "Voice input");
        const message = error.name === "NotAllowedError" || error.name === "SecurityError"
            ? "Microphone permission was denied. Allow microphone access and try again."
            : error.name === "NotFoundError"
                ? "No microphone was found. Connect a microphone and try again."
                : error.name === "NotReadableError"
                    ? "The microphone is busy or unavailable. Close other apps using it and try again."
                    : "Could not start microphone recording. Check microphone access and try again.";
        showToast(message, "error");
    });
}

async function copyText(value) {
    try {
        await navigator.clipboard.writeText(value);
        showToast("Copied to clipboard.", "success");
    } catch (error) {
        showToast("Clipboard access is unavailable in this browser context.", "error");
    }
}


function stopVoiceRecognition() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
}


async function transcribeAudio(blob) {
    const mime = blob.type || "audio/webm";
    const extension = mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("file", blob, `voice.${extension}`);
    form.append("language", getRecognitionLanguage());
    showToast("Transcribing locally…", "info");
    try {
        const response = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Local transcription failed.");
        const input = $("questionInput");
        if (input) { input.value = data.text || ""; input.focus(); }
        showToast("Voice input ready.", "success");
    } catch (error) {
        showToast(error.message, "error");
    }
}


// ============================================================
// TEXT TO SPEECH
// ============================================================

function stopAllSpeech() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    if (currentSpeechButton) {
        currentSpeechButton.textContent = "🔊 Listen";
        currentSpeechButton.classList.remove("speaking");
        currentSpeechButton.disabled = false;
        currentSpeechButton = null;
    }
    isSpeaking = false;
    currentSpeech = null;
}


function toggleSpeech(
    text,
    button
) {
    if (isSpeaking) {
        const wasActive = (currentSpeechButton === button);
        stopAllSpeech();
        if (wasActive) {
            return;
        }
    }

    speakText(
        text,
        button
    );
}


function findVoiceForLanguage(langCode) {
    if (!window.speechSynthesis) return null;
    const voices = window.speechSynthesis.getVoices() || [];
    const prefix = langCode.split("-")[0].toLowerCase();
    return voices.find(v => {
        const vl = (v.lang || "").toLowerCase();
        return vl === langCode.toLowerCase() || vl.startsWith(prefix) || vl.includes(prefix);
    }) || null;
}


function speakText(
    text,
    button
) {
    stopAllSpeech();

    const language =
        getSpeechSynthesisLanguage();

    const hasTeluguChars = /[\u0C00-\u0C7F]/.test(text || "");
    const isTelugu = language === "te-IN" || hasTeluguChars;

    if (isTelugu) {
        speakTeluguOffline(text, button);
        return;
    }

    if (
        !window.speechSynthesis
    ) {
        showToast(
            "Text-to-speech is not supported.",
            "error"
        );
        return;
    }

    const utterance =
        new SpeechSynthesisUtterance(
            text
        );

    utterance.lang =
        language;

    utterance.rate = 0.95;

    utterance.pitch = 1;

    const matchedVoice = findVoiceForLanguage(language);
    if (matchedVoice) {
        utterance.voice = matchedVoice;
    }

    currentSpeech =
        utterance;
    currentSpeechButton =
        button;
    isSpeaking = true;

    if (button) {
        button.textContent =
            "⏹ Stop";
        button.classList.add(
            "speaking"
        );
    }

    utterance.onend = () => {
        if (currentSpeech === utterance) {
            isSpeaking = false;
            currentSpeech = null;
            if (button) {
                button.textContent =
                    "🔊 Listen";
                button.classList.remove(
                    "speaking"
                );
            }
            if (currentSpeechButton === button) {
                currentSpeechButton = null;
            }
        }
    };

    utterance.onerror = () => {
        if (currentSpeech === utterance) {
            isSpeaking = false;
            currentSpeech = null;
            if (button) {
                button.textContent =
                    "🔊 Listen";
                button.classList.remove(
                    "speaking"
                );
            }
            if (currentSpeechButton === button) {
                currentSpeechButton = null;
            }
        }
    };

    window.speechSynthesis.speak(
        utterance
    );
}


let offlineTeluguSpeechRequestId = 0;

async function speakTeluguOffline(text, button) {
    if (!text || !text.trim()) {
        showToast("No text available to read.", "warning");
        return;
    }

    const thisRequestId = ++offlineTeluguSpeechRequestId;

    if (isSpeaking) {
        stopAllSpeech();
    }

    if (button) {
        button.textContent = "⏳ Loading...";
        button.disabled = true;
    }
    currentSpeechButton = button;

    try {
        const response = await fetch("/api/tts", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                language: "Telugu"
            })
        });

        if (thisRequestId !== offlineTeluguSpeechRequestId) {
            if (button) {
                button.disabled = false;
                button.textContent = "🔊 Listen";
                button.classList.remove("speaking");
            }
            return;
        }

        if (!response.ok) {
            let errorMsg = "Offline Telugu voice is not installed or available on this system. Run the offline Telugu TTS model setup.";
            try {
                const errData = await response.json();
                if (errData.detail) errorMsg = errData.detail;
            } catch (_) {}
            throw new Error(errorMsg);
        }

        const audioBlob = await response.blob();
        if (thisRequestId !== offlineTeluguSpeechRequestId) {
            if (button) {
                button.disabled = false;
                button.textContent = "🔊 Listen";
                button.classList.remove("speaking");
            }
            return;
        }

        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        currentAudio = audio;
        isSpeaking = true;

        if (button) {
            button.disabled = false;
            button.textContent = "⏹ Stop";
            button.classList.add("speaking");
        }

        audio.onended = () => {
            if (currentAudio === audio) {
                isSpeaking = false;
                currentAudio = null;
                URL.revokeObjectURL(audioUrl);
                if (button) {
                    button.textContent = "🔊 Listen";
                    button.classList.remove("speaking");
                }
                if (currentSpeechButton === button) {
                    currentSpeechButton = null;
                }
            }
        };

        audio.onerror = () => {
            if (currentAudio === audio) {
                isSpeaking = false;
                currentAudio = null;
                URL.revokeObjectURL(audioUrl);
                if (button) {
                    button.textContent = "🔊 Listen";
                    button.classList.remove("speaking");
                }
                if (currentSpeechButton === button) {
                    currentSpeechButton = null;
                }
                showToast("Failed to play Telugu audio.", "error");
            }
        };

        await audio.play();
    } catch (error) {
        if (thisRequestId === offlineTeluguSpeechRequestId) {
            if (button) {
                button.disabled = false;
                button.textContent = "🔊 Listen";
                button.classList.remove("speaking");
            }
            if (currentSpeechButton === button) {
                currentSpeechButton = null;
            }
            isSpeaking = false;
            currentAudio = null;
            showToast(error.message, "warning");
        }
    }
}


function getSpeechSynthesisLanguage() {

    const language =
        $("answerLanguage");

    const selected =
        language
            ? language.value
            : "English";

    if (selected === "Telugu") {
        return "te-IN";
    }

    if (selected === "Hindi") {
        return "hi-IN";
    }

    return "en-IN";
}


// ============================================================
// SIDEBAR
// ============================================================

function setupSidebar() {

    const menuButton =
        $("mobileMenuBtn");

    if (menuButton) {

        menuButton.addEventListener(
            "click",
            toggleMobileSidebar
        );

    }

    const clearButton =
        $("clearAllPdfBtn");

    if (clearButton) {

        clearButton.addEventListener(
            "click",
            clearAllDocuments
        );

    }

    const summarizeButton =
        $("summarizePdfBtn");

    if (summarizeButton) {

        summarizeButton.addEventListener(
            "click",
            summarizePDF
        );

    }

    const newChatButton =
        $("newChatBtn");

    if (newChatButton) {

        newChatButton.addEventListener(
            "click",
            startNewChat
        );

    }

}


function toggleMobileSidebar() {

    const sidebar =
        query(".sidebar");

    if (sidebar) {

        sidebar.classList.toggle(
            "open"
        );

    }
}


function closeMobileSidebar() {

    const sidebar =
        query(".sidebar");

    if (sidebar) {

        sidebar.classList.remove(
            "open"
        );

    }
}


// ============================================================
// KEYBOARD
// ============================================================

function setupKeyboard() {

    document.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Escape"
            ) {

                closeMobileSidebar();

                if (isRecording) {
                    stopVoiceRecognition();
                }

                if (isSpeaking) {
                    stopAllSpeech();
                }

            }

        }
    );
}


// ============================================================
// SCROLL CHAT
// ============================================================

function scrollChatToBottom() {

    const messages =
        $("chatMessages");

    if (!messages) {
        return;
    }

    requestAnimationFrame(
        () => {

            messages.scrollTop =
                messages.scrollHeight;

            const pageArea =
                query(".page-area");

            if (pageArea) {

                pageArea.scrollTop =
                    pageArea.scrollHeight;

            }

        }
    );
}


// ============================================================
// TOAST
// ============================================================

function showToast(
    message,
    type = "info"
) {

    const container =
        $("toastContainer");

    if (!container) {
        return;
    }

    const toast =
        document.createElement("div");

    toast.className =
        `toast toast-${type}`;

    toast.textContent =
        message;

    container.appendChild(
        toast
    );

    requestAnimationFrame(
        () => {

            toast.classList.add(
                "show"
            );

        }
    );

    setTimeout(
        () => {

            toast.classList.remove(
                "show"
            );

            setTimeout(
                () => {

                    toast.remove();

                },
                300
            );

        },
        2800
    );
}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(
    value
) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


// ============================================================
// GLOBAL FUNCTIONS
// ============================================================

window.navigateTo =
    navigateTo;

window.removeDocument =
    removeDocument;

window.clearAllDocuments =
    clearAllDocuments;

window.summarizePDF =
    summarizePDF;
window.goHome = goHome;
window.newChat = newChat;
window.showQuiz = showQuiz;
window.backFromQuiz = backFromQuiz;
window.onQuizDocChange = onQuizDocChange;
window.onGenerateQuizClick = onGenerateQuizClick;
window.updateQuizDocSelector = updateQuizDocSelector;
window.showSettings = showSettings;
window.showHelp = showHelp;
window.clearAllPDFs = clearAllPDFs;
window.syncSettingsLanguage = syncSettingsLanguage;
window.changeLanguage = changeLanguage;
window.showUploadedPDFs = showUploadedPDFs;

window.startNewChat =
    startNewChat;

window.sendQuestion =
    sendQuestion;
