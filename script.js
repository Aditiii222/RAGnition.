/**
 * RAGnition — Intelligent Document Retrieval and Question Answering
 * Core Pipeline Implementation: Lexical DPR Baseline, Sentence Grounding, CRAG Correction
 */

(function () {
  'use strict';

  // ==========================================
  // 1. DEFAULT EDUCATIONAL CORPUS
  // ==========================================
  const DEFAULT_DOCUMENTS = [
    {
      id: 'doc-rag',
      title: 'Retrieval-Augmented Generation (RAG)',
      isDefault: true,
      content: `Retrieval-Augmented Generation (RAG) is an architectural framework in artificial intelligence that combines information retrieval systems with generative language models. By retrieving authoritative context from external knowledge bases prior to generating a response, RAG substantially mitigates hallucinations, circumvents static knowledge cutoff boundaries, and produces verifiable source citations.

The retrieval phase in a RAG pipeline acts as the primary knowledge filter. Incoming user queries are normalized, tokenized, and mapped against an indexed document corpus. Retrieval strategies span sparse lexical techniques such as BM25 and TF-IDF to dense passage retrieval using dual-encoder neural representations. Document chunking partitions extensive documents into paragraph-aware passages, preserving contextual coherence while fitting within strict context window constraints.

The generation phase utilizes the retrieved top-ranking evidence passages to condition the model's synthesis. The language model generates answers grounded directly on the extracted evidence, ensuring factual fidelity. Each assertion within the generated text can be mapped back to identifiable source chunks, providing transparent attribution and enabling human verifiability in mission-critical applications.

Real-world advantages of RAG include cost efficiency and agile knowledge updates. Organizations can update internal guidelines, research findings, or policy documents instantaneously by updating the document index without requiring expensive and slow model fine-tuning or retraining. Furthermore, strict access control can be enforced at the retrieval layer to safeguard proprietary data.`
    },
    {
      id: 'doc-aiml',
      title: 'Artificial Intelligence and Machine Learning',
      isDefault: true,
      content: `Artificial intelligence (AI) and machine learning (ML) represent a computational paradigm where algorithms learn patterns, representations, and decision rules directly from data rather than following rigid procedural instructions. The primary learning paradigms include supervised learning from labeled datasets, unsupervised clustering and representation learning, and reinforcement learning governed by reward feedback.

Deep learning architectures rely on multilayer neural networks that iteratively adjust weight parameters via backpropagation and gradient descent optimization. As networks deepen, representations transition from low-level sensory features to abstract semantic concepts. Over the last decade, deep learning has revolutionized computer vision, speech synthesis, predictive modeling, and robotics.

Natural Language Processing (NLP) experienced a foundational breakthrough with the introduction of the Transformer architecture and self-attention mechanisms. Self-attention enables models to weigh relational significance between distant tokens across a sequence simultaneously. This replaced legacy recurrent neural networks and enabled massive scale foundation models capable of contextual text comprehension and multilingual translation.

Rigorous evaluation of machine learning models relies on quantitative metrics such as precision, recall, F1-score, and area under the ROC curve (AUC). In text generation and retrieval, evaluation incorporates mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), and perplexity. Ethical machine learning demands transparency regarding dataset biases, explainability of predictions, and robust privacy preservation.`
    },
    {
      id: 'doc-climate',
      title: 'Climate Change: Causes, Mechanisms, and Impacts',
      isDefault: true,
      content: `Climate change refers to persistent long-term shifts in global atmospheric temperatures and regional weather patterns. While geological climate variations have occurred naturally across millennia, contemporary warming is overwhelmingly driven by anthropogenic greenhouse gas emissions that alter Earth's atmospheric radiative equilibrium.

The greenhouse effect is the fundamental physical mechanism governing planetary temperatures. Solar radiation penetrates Earth's atmosphere, warming the terrestrial and oceanic surfaces. The warmed surface re-radiates energy as infrared radiation, which is absorbed and re-emitted by greenhouse gases including carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). This trapping of thermal energy warms the lower troposphere.

The principal anthropogenic drivers of elevated greenhouse gas concentrations are the combustion of fossil fuels (coal, petroleum, and natural gas) for electricity generation, industrial manufacturing, and transportation. Secondary major contributors include deforestation, which depletes crucial terrestrial carbon sinks, and intensive industrial agriculture, which releases extensive quantities of methane from livestock and nitrous oxide from nitrogen-based fertilizers.

Observable global consequences include elevated global mean surface temperatures, acceleration of polar ice sheet and glacier melt, rising sea levels threatening coastal ecosystems, ocean acidification caused by excessive oceanic absorption of carbon dioxide, and increased frequency and severity of extreme meteorological events such as droughts, heatwaves, and tropical cyclones. Mitigation necessitates global decarbonization, rapid transition to renewable energy sources, and sustainable land management.`
    }
  ];

  // ==========================================
  // 2. STOP WORDS & EXPANSION LEXICON
  // ==========================================
  const STOP_WORDS = new Set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'can\'t', 'cannot', 'could', 'couldn\'t',
    'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each', 'few', 'for', 'from', 'further',
    'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
    'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s',
    'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
    'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such',
    'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they',
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very',
    'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t',
    'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
  ]);

  // Deterministic expansion vocabulary for CRAG
  const SYNONYM_MAP = {
    'rag': ['retrieval', 'augmented', 'generation', 'external', 'knowledge'],
    'retrieval': ['retrieve', 'search', 'passages', 'corpus', 'index', 'evidence'],
    'generation': ['generative', 'synthesis', 'language', 'models'],
    'improve': ['enhances', 'mitigates', 'benefits', 'accuracy', 'fidelity'],
    'improves': ['enhances', 'mitigates', 'benefits', 'accuracy'],
    'improving': ['enhancing', 'mitigating', 'benefiting'],
    'climate': ['global', 'warming', 'greenhouse', 'atmospheric', 'temperatures'],
    'causes': ['drivers', 'mechanisms', 'emissions', 'combustion', 'sources'],
    'cause': ['driver', 'mechanism', 'emission', 'combustion'],
    'machine': ['artificial', 'intelligence', 'deep', 'learning', 'algorithms'],
    'ai': ['artificial', 'intelligence', 'learning', 'models'],
    'ml': ['machine', 'learning', 'algorithms', 'supervised'],
    'spaceflight': ['orbital', 'lunar', 'spacecraft', 'aerospace', 'missions']
  };

  // ==========================================
  // 3. APPLICATION STATE
  // ==========================================
  const state = {
    documents: [],
    chunks: [],
    selectedDocId: null,
    currentResults: null
  };

  // ==========================================
  // 4. CHUNKING & INDEXING PIPELINE
  // ==========================================

  /**
   * Splits document text into paragraph-aware chunks with unique IDs.
   */
  function chunkDocument(doc) {
    const rawParagraphs = doc.content
      .split(/\n\s*\n/)
      .map(p => p.trim())
      .filter(p => p.length > 0);

    const docChunks = [];

    rawParagraphs.forEach((para, idx) => {
      const words = para.split(/\s+/);
      const chunkId = `${doc.id.toUpperCase()}-CHK${idx + 1}`;
      
      // Extract individual sentences for fine-grained grounding
      const sentences = para
        .split(/(?<=[.!?])\s+/)
        .map(s => s.trim())
        .filter(s => s.length > 5);

      docChunks.push({
        id: chunkId,
        docId: doc.id,
        docTitle: doc.title,
        text: para,
        wordCount: words.length,
        sentences: sentences
      });
    });

    return docChunks;
  }

  /**
   * Rebuilds the global chunk index across all active documents.
   */
  function rebuildIndex() {
    state.chunks = [];
    state.documents.forEach(doc => {
      const docChunks = chunkDocument(doc);
      state.chunks.push(...docChunks);
    });
    updateCorpusBadges();
  }

  // ==========================================
  // 5. QUERY PROCESSING & LEXICAL RETRIEVAL
  // ==========================================

  /**
   * Normalizes and tokenizes text, removing stop words.
   */
  function processQuery(text) {
    if (!text || typeof text !== 'string') return [];
    
    // Lowercase and strip non-alphanumeric characters
    const rawTokens = text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .map(t => t.trim())
      .filter(t => t.length > 1);

    const filteredTokens = rawTokens.filter(t => !STOP_WORDS.has(t));
    return filteredTokens.length > 0 ? filteredTokens : rawTokens;
  }

  /**
   * Helper to normalize word for stemming (removes common suffixes)
   */
  function stemWord(word) {
    if (!word || word.length < 4) return word;
    return word
      .replace(/(ing|edly|ingly|ed|es|s|tion|ment|al|ic|ive)$/i, '')
      .trim();
  }

  /**
   * Computes BM25/IDF Lexical Retrieval over all chunks.
   * Labeled: Lexical Retrieval — Simplified DPR-Inspired Baseline
   */
  function retrieveEvidence(queryTokens, candidateChunks = state.chunks) {
    if (queryTokens.length === 0 || candidateChunks.length === 0) {
      return [];
    }

    const totalChunks = candidateChunks.length;
    const avgChunkLength = candidateChunks.reduce((acc, c) => acc + c.wordCount, 0) / totalChunks;
    const k1 = 1.5;
    const b = 0.75;

    // Stem query tokens
    const stemmedQueryTokens = queryTokens.map(t => ({
      raw: t,
      stem: stemWord(t)
    }));

    // 1. Calculate Document Frequency (DF) for each query token across chunks
    const tokenDF = {};
    stemmedQueryTokens.forEach(({ raw, stem }) => {
      let count = 0;
      candidateChunks.forEach(chunk => {
        const chunkWords = chunk.text
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, ' ')
          .split(/\s+/)
          .filter(w => w.length > 0);

        const hasMatch = chunkWords.some(w => w === raw || (stem.length >= 4 && stemWord(w) === stem));
        if (hasMatch) {
          count++;
        }
      });
      tokenDF[raw] = count;
    });

    // 2. Score each chunk using BM25 formula
    const scoredChunks = candidateChunks.map(chunk => {
      const chunkWords = chunk.text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 0);
      
      const chunkWordCount = chunkWords.length;

      // Word frequency map within this chunk
      const tfMap = {};
      chunkWords.forEach(w => {
        tfMap[w] = (tfMap[w] || 0) + 1;
      });

      let bm25Score = 0;
      const matchedTerms = [];

      stemmedQueryTokens.forEach(({ raw, stem }) => {
        const df = tokenDF[raw] || 0;
        if (df === 0) {
          // Term does not exist anywhere in the corpus
          return;
        }

        // Exact TF
        let tf = tfMap[raw] || 0;

        // Stemmed fallback
        if (tf === 0 && stem.length >= 4) {
          for (const w of chunkWords) {
            if (stemWord(w) === stem) {
              tf += 0.8; // stem match
            }
          }
        }

        if (tf > 0) {
          matchedTerms.push(raw);
          // Robertson-Spärck Jones IDF
          const idf = Math.log(1 + (totalChunks - df + 0.5) / (df + 0.5));
          const safeIdf = Math.max(idf, 0.1);
          const numerator = tf * (k1 + 1);
          const denominator = tf + k1 * (1 - b + b * (chunkWordCount / avgChunkLength));
          bm25Score += safeIdf * (numerator / denominator);
        }
      });

      return {
        chunk: chunk,
        rawScore: parseFloat(bm25Score.toFixed(3)),
        matchedTerms: Array.from(new Set(matchedTerms)),
        matchCoverage: matchedTerms.length / queryTokens.length
      };
    });

    // Sort descending by rawScore
    scoredChunks.sort((a, b) => b.rawScore - a.rawScore);

    // Return top 3 chunks
    return scoredChunks.slice(0, 3);
  }

  // ==========================================
  // 6. CRAG-INSPIRED RETRIEVAL CORRECTION
  // ==========================================

  /**
   * Evaluates evidence quality and applies deterministic query expansion if weak.
   */
  function evaluateAndCorrectRetrieval(originalQuery, initialTokens, initialResults) {
    const topResult = initialResults[0];
    const topScore = topResult ? topResult.rawScore : 0;
    const topCoverage = topResult ? topResult.matchCoverage : 0;

    // Transparent Relevance Thresholds:
    // Score >= 3.0 and coverage >= 0.35 -> Relevant
    // Score >= 1.0 and coverage >= 0.20 -> Partial
    // Otherwise -> Insufficient
    let initialVerdict = 'insufficient';
    if (topScore >= 2.5 && topCoverage >= 0.35) {
      initialVerdict = 'relevant';
    } else if (topScore >= 0.8 && topCoverage >= 0.15) {
      initialVerdict = 'partial';
    }

    // If initial retrieval is already relevant and confident, return immediately
    if (initialVerdict === 'relevant') {
      return {
        verdict: 'relevant',
        cragTriggered: false,
        results: initialResults,
        usedQuery: originalQuery,
        effectiveTokens: initialTokens,
        message: 'Relevant Evidence Found: High confidence match with corpus passages.'
      };
    }

    // Step 1: CRAG Correction Triggered — Expand query terms deterministically
    const expandedTokens = [...initialTokens];
    initialTokens.forEach(t => {
      if (SYNONYM_MAP[t]) {
        SYNONYM_MAP[t].forEach(syn => {
          if (!expandedTokens.includes(syn)) {
            expandedTokens.push(syn);
          }
        });
      }
    });

    // Step 2: Second retrieval pass using expanded query terms
    const secondResults = retrieveEvidence(expandedTokens);
    const secondTop = secondResults[0];
    const secondScore = secondTop ? secondTop.rawScore : 0;
    const secondCoverage = secondTop ? secondTop.matchCoverage : 0;

    // Step 3: Compare passes
    const scoreDiff = parseFloat((secondScore - topScore).toFixed(3));
    const evidenceImproved = secondScore > topScore && secondResults.length > 0;

    let finalVerdict = 'insufficient';
    if (secondScore >= 2.5 && secondCoverage >= 0.25) {
      finalVerdict = 'relevant';
    } else if (secondScore >= 0.8 && secondCoverage >= 0.15) {
      finalVerdict = 'partial';
    }

    return {
      verdict: finalVerdict,
      cragTriggered: true,
      evidenceImproved: evidenceImproved,
      originalScore: topScore,
      correctedScore: secondScore,
      scoreDiff: scoreDiff,
      originalTokens: initialTokens,
      expandedTokens: expandedTokens,
      results: evidenceImproved ? secondResults : initialResults,
      usedQuery: originalQuery,
      effectiveTokens: expandedTokens,
      message: evidenceImproved 
        ? `CRAG Correction Applied: Query expansion improved score from ${topScore} to ${secondScore}.`
        : `CRAG Correction Evaluated: No significant evidence gain (Delta: ${scoreDiff}). Corpus lacks sufficient support.`
    };
  }

  // ==========================================
  // 7. EVIDENCE-GROUNDED ANSWER SYNTHESIS
  // ==========================================

  /**
   * Synthesizes an evidence-grounded answer by extracting and ranking key sentences from retrieved chunks.
   * Labeled: Local Evidence-Grounded RAG Prototype
   */
  function synthesizeAnswer(queryTokens, evaluatedRetrieval) {
    const { verdict, results } = evaluatedRetrieval;

    // Refusal handling for insufficient evidence
    if (verdict === 'insufficient' || results.length === 0 || results[0].rawScore < 0.3) {
      return {
        answerHtml: `<p style="color: var(--danger-text); font-weight: 500;">
          <strong>Insufficient Evidence in Corpus:</strong> RAGnition examined all active documents but could not find sufficient authoritative evidence to answer this question reliably.
        </p>
        <p style="margin-top: 8px; font-size: 0.88rem; color: var(--text-secondary);">
          To prevent hallucination, the system strictly declines to synthesize an unsupported answer. You can expand the knowledge corpus by uploading or pasting a relevant document in the <em>Document Library</em>.
        </p>`,
        citations: [],
        verdict: 'insufficient'
      };
    }

    // Collect candidate sentences from the top chunks
    const candidateSentences = [];
    results.forEach((item, chunkRank) => {
      const chunk = item.chunk;
      const chunkScoreWeight = item.rawScore * 0.4; // Prioritize sentences from higher scoring chunks

      chunk.sentences.forEach((sentence, sIdx) => {
        const sentenceWords = sentence
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, ' ')
          .split(/\s+/)
          .filter(w => w.length > 0);

        let sentenceScore = chunkScoreWeight;
        const matchedTokens = [];

        queryTokens.forEach(token => {
          const stem = stemWord(token);
          const hasWord = sentenceWords.some(w => w === token || (stem.length >= 4 && stemWord(w) === stem));
          if (hasWord) {
            matchedTokens.push(token);
            sentenceScore += 2.0;
          }
        });

        // Boost first sentence of the top chunk
        if (chunkRank === 0 && sIdx === 0) sentenceScore += 1.0;
        if (sIdx === 0) sentenceScore += 0.3;

        if (matchedTokens.length > 0 || chunkRank === 0) {
          candidateSentences.push({
            text: sentence,
            score: sentenceScore,
            chunkId: chunk.id,
            docTitle: chunk.docTitle,
            matchedCount: matchedTokens.length
          });
        }
      });
    });

    // Sort candidate sentences by relevance score
    candidateSentences.sort((a, b) => b.score - a.score);

    // Pick top 2 to 3 non-redundant sentences
    const selectedSentences = [];
    const usedText = new Set();
    const citations = [];

    for (const cand of candidateSentences) {
      if (selectedSentences.length >= 3) break;
      if (!usedText.has(cand.text)) {
        usedText.add(cand.text);
        selectedSentences.push(cand);
        citations.push({
          chunkId: cand.chunkId,
          docTitle: cand.docTitle
        });
      }
    }

    // Fallback if no specific sentence had high overlap
    if (selectedSentences.length === 0 && results[0]) {
      const topChunk = results[0].chunk;
      const firstSentence = topChunk.sentences[0] || topChunk.text;
      selectedSentences.push({
        text: firstSentence,
        chunkId: topChunk.id,
        docTitle: topChunk.docTitle
      });
      citations.push({ chunkId: topChunk.id, docTitle: topChunk.docTitle });
    }

    // Build synthesized answer HTML with interactive clickable citation badges
    let answerHtml = '';
    selectedSentences.forEach((s) => {
      const citationHtml = `<button class="citation-pill" data-chunk-id="${s.chunkId}" title="Inspect source chunk in ${s.docTitle}">
        [${s.chunkId}]
      </button>`;
      
      answerHtml += `<span>${escapeHtml(s.text)}</span> ${citationHtml} `;
    });

    return {
      answerHtml: `<p>${answerHtml.trim()}</p>`,
      citations: citations,
      verdict: verdict
    };
  }

  // ==========================================
  // 8. STORAGE & PERSISTENCE
  // ==========================================
  const STORAGE_KEY = 'ragnition_custom_docs_v2';

  function loadDocumentsFromStorage() {
    state.documents = [...DEFAULT_DOCUMENTS];
    try {
      const stored = localStorage.getItem(STORAGE_KEY) || localStorage.getItem('documind_custom_docs_v1');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          parsed.forEach(customDoc => {
            if (customDoc && customDoc.id && customDoc.title && customDoc.content) {
              state.documents.push(customDoc);
            }
          });
        }
      }
    } catch (e) {
      console.warn('LocalStorage unavailable or parse error:', e);
    }
    rebuildIndex();
  }

  function saveCustomDocumentsToStorage() {
    try {
      const customDocs = state.documents.filter(d => !d.isDefault);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(customDocs));
    } catch (e) {
      console.warn('Failed to save to localStorage:', e);
    }
  }

  // ==========================================
  // 9. UI RENDERING & EVENT HANDLERS
  // ==========================================

  function updateCorpusBadges() {
    const docCountEl = document.getElementById('corpusDocCount');
    const chunkCountEl = document.getElementById('corpusChunkCount');
    if (docCountEl) docCountEl.textContent = `${state.documents.length} Docs`;
    if (chunkCountEl) chunkCountEl.textContent = `${state.chunks.length} Chunks`;

    // Dashboard Statistics Updates
    const statDoc = document.getElementById('statDocCount');
    const statChunk = document.getElementById('statChunkCount');
    const statWord = document.getElementById('statWordCount');

    if (statDoc) statDoc.textContent = state.documents.length;
    if (statChunk) statChunk.textContent = state.chunks.length;
    if (statWord) {
      const totalWords = state.documents.reduce((acc, d) => {
        return acc + d.content.split(/\s+/).filter(w => w.length > 0).length;
      }, 0);
      statWord.textContent = `~${totalWords.toLocaleString()}`;
    }
  }

  function renderDocumentList() {
    const container = document.getElementById('docListContainer');
    if (!container) return;

    container.innerHTML = '';
    state.documents.forEach(doc => {
      const docChunks = state.chunks.filter(c => c.docId === doc.id);
      const words = doc.content.split(/\s+/).length;

      const item = document.createElement('div');
      item.className = `doc-item ${state.selectedDocId === doc.id ? 'active' : ''}`;
      item.dataset.docId = doc.id;

      item.innerHTML = `
        <div class="doc-item-header">
          <div class="doc-item-title">${escapeHtml(doc.title)}</div>
          <span class="doc-type-pill">${doc.isDefault ? 'Default' : 'Custom'}</span>
        </div>
        <div class="doc-item-meta">
          <span>${docChunks.length} Chunks</span>
          <span>&bull;</span>
          <span>~${words} words</span>
          ${!doc.isDefault ? `<button class="btn btn-danger-outline btn-sm delete-doc-btn" data-doc-id="${doc.id}" style="margin-left: auto; padding: 2px 6px;">Delete</button>` : ''}
        </div>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.closest('.delete-doc-btn')) {
          const toDeleteId = e.target.closest('.delete-doc-btn').dataset.docId;
          deleteCustomDocument(toDeleteId);
          return;
        }
        selectDocument(doc.id);
      });

      container.appendChild(item);
    });
  }

  function selectDocument(docId) {
    state.selectedDocId = docId;
    renderDocumentList();

    const doc = state.documents.find(d => d.id === docId);
    const previewTitle = document.getElementById('previewDocTitle');
    const previewMeta = document.getElementById('previewDocMeta');
    const previewBadge = document.getElementById('previewChunkCountBadge');
    const previewContent = document.getElementById('docPreviewContent');

    if (!doc) {
      if (previewTitle) previewTitle.textContent = 'Select a document to inspect';
      if (previewMeta) previewMeta.textContent = 'Document overview and paragraph-aware chunk breakdown';
      if (previewBadge) previewBadge.textContent = '0 Chunks';
      if (previewContent) previewContent.innerHTML = '<div class="empty-state"><p class="empty-text">Select a document from the left panel.</p></div>';
      return;
    }

    const docChunks = state.chunks.filter(c => c.docId === doc.id);

    if (previewTitle) previewTitle.textContent = doc.title;
    if (previewMeta) previewMeta.textContent = `${doc.isDefault ? 'Preloaded Educational Reference' : 'User Uploaded Text'} &bull; ${doc.content.length} characters`;
    if (previewBadge) previewBadge.textContent = `${docChunks.length} Chunks`;

    if (previewContent) {
      let chunksHtml = '';
      docChunks.forEach(chunk => {
        chunksHtml += `
          <div class="doc-chunk-preview">
            <div class="doc-chunk-header">${chunk.id} (${chunk.wordCount} words)</div>
            <div>${escapeHtml(chunk.text)}</div>
          </div>
        `;
      });
      previewContent.innerHTML = chunksHtml;
    }
  }

  function deleteCustomDocument(docId) {
    state.documents = state.documents.filter(d => d.id !== docId);
    saveCustomDocumentsToStorage();
    rebuildIndex();
    if (state.selectedDocId === docId) {
      state.selectedDocId = state.documents[0] ? state.documents[0].id : null;
    }
    renderDocumentList();
    if (state.selectedDocId) {
      selectDocument(state.selectedDocId);
    }
  }

  function handleAddDocument(title, content) {
    const trimmedTitle = title ? title.trim() : '';
    const trimmedContent = content ? content.trim() : '';

    if (!trimmedTitle || !trimmedContent) {
      alert('Please provide both a document title and content.');
      return false;
    }

    const newId = `doc-custom-${Date.now()}`;
    const newDoc = {
      id: newId,
      title: trimmedTitle,
      content: trimmedContent,
      isDefault: false
    };

    state.documents.push(newDoc);
    saveCustomDocumentsToStorage();
    rebuildIndex();
    selectDocument(newId);
    return true;
  }

  function highlightEvidenceCard(chunkId) {
    const cards = document.querySelectorAll('.evidence-card');
    cards.forEach(c => c.classList.remove('highlighted'));

    const target = document.querySelector(`.evidence-card[data-chunk-id="${chunkId}"]`);
    if (target) {
      target.classList.add('highlighted');
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function openChunkModal(chunkId) {
    const chunk = state.chunks.find(c => c.id === chunkId);
    if (!chunk) return;

    const modal = document.getElementById('chunkModal');
    const modalChunkId = document.getElementById('modalChunkId');
    const modalDocTitle = document.getElementById('modalDocTitle');
    const modalBody = document.getElementById('modalChunkBody');

    if (modal && modalChunkId && modalDocTitle && modalBody) {
      modalChunkId.textContent = `Chunk ${chunk.id}`;
      modalDocTitle.textContent = `Source: ${chunk.docTitle} (${chunk.wordCount} words)`;
      modalBody.textContent = chunk.text;
      modal.style.display = 'flex';
    }
  }

  function closeChunkModal() {
    const modal = document.getElementById('chunkModal');
    if (modal) modal.style.display = 'none';
  }

  /**
   * Main Question Answering Execution Handler
   */
  function handleAskQuestion(queryText) {
    const question = queryText ? queryText.trim() : '';
    if (!question) {
      alert('Please enter a question to search the document corpus.');
      return;
    }

    const startTime = performance.now();

    // Step 1: Query Ingestion & Processing
    updatePipelineTrack(1);
    const initialTokens = processQuery(question);

    // Step 2: Lexical Evidence Retrieval (BM25 baseline)
    updatePipelineTrack(2);
    const initialResults = retrieveEvidence(initialTokens);

    // Step 3: CRAG Relevance Assessment & Correction
    updatePipelineTrack(3);
    const evaluation = evaluateAndCorrectRetrieval(question, initialTokens, initialResults);

    // Step 4: Evidence Grounding & Answer Synthesis
    updatePipelineTrack(4);
    const answerResult = synthesizeAnswer(evaluation.effectiveTokens, evaluation);
    const elapsedMs = Math.round(performance.now() - startTime);

    // Store state
    state.currentResults = {
      question: question,
      evaluation: evaluation,
      answerResult: answerResult,
      elapsedMs: elapsedMs
    };

    // Update latency stat
    const statLatency = document.getElementById('statLatency');
    if (statLatency) {
      statLatency.textContent = `${elapsedMs}ms`;
    }

    // Render Answer and CRAG status
    renderAnswerOutput(state.currentResults);

    // Render Retrieved Evidence Cards
    renderEvidenceList(evaluation.results, evaluation.effectiveTokens);
  }

  function updatePipelineTrack(activeStep) {
    for (let i = 1; i <= 4; i++) {
      const stepEl = document.getElementById(`pipeStep${i}`);
      if (!stepEl) continue;
      stepEl.classList.remove('active', 'success');
      if (i < activeStep) {
        stepEl.classList.add('success');
      } else if (i === activeStep) {
        stepEl.classList.add('active');
      }
    }
  }

  function renderAnswerOutput(currentResults) {
    const { evaluation, answerResult, elapsedMs } = currentResults;
    const verdictBadge = document.getElementById('verdictBadge');
    const cragContainer = document.getElementById('cragNoticeContainer');
    const answerBody = document.getElementById('answerBody');
    const metaInfo = document.getElementById('pipelineMetaInfo');

    // Verdict Badge update
    if (verdictBadge) {
      verdictBadge.className = `verdict-badge ${answerResult.verdict}`;
      if (answerResult.verdict === 'relevant') {
        verdictBadge.innerHTML = '<span class="status-dot" style="background:#10b981;"></span> Relevant Evidence Found';
      } else if (answerResult.verdict === 'partial') {
        verdictBadge.innerHTML = '<span class="status-dot" style="background:#f59e0b;"></span> Partial Evidence Found';
      } else {
        verdictBadge.innerHTML = '<span class="status-dot" style="background:#ef4444;"></span> Insufficient Evidence';
      }
    }

    // CRAG Notice Box
    if (cragContainer) {
      if (evaluation.cragTriggered) {
        cragContainer.style.display = 'block';
        cragContainer.innerHTML = `
          <div class="crag-box">
            <div class="crag-box-header">
              <span>CRAG-Inspired Retrieval Correction</span>
              <span style="font-size:0.75rem;">Score: ${evaluation.originalScore} &rarr; ${evaluation.correctedScore} (Delta: ${evaluation.scoreDiff >= 0 ? '+' : ''}${evaluation.scoreDiff})</span>
            </div>
            <div class="crag-box-body">
              ${escapeHtml(evaluation.message)}
              <div class="crag-tags">
                <span style="font-size:0.75rem; color:#86198f; font-weight:600;">Expanded Query Terms:</span>
                ${evaluation.expandedTokens.map(t => `<span class="crag-tag">${escapeHtml(t)}</span>`).join('')}
              </div>
            </div>
          </div>
        `;
      } else {
        cragContainer.style.display = 'none';
        cragContainer.innerHTML = '';
      }
    }

    // Answer Body
    if (answerBody) {
      answerBody.innerHTML = answerResult.answerHtml;

      // Attach click listeners to generated citation pills
      answerBody.querySelectorAll('.citation-pill').forEach(pill => {
        pill.addEventListener('click', (e) => {
          e.preventDefault();
          const chunkId = pill.dataset.chunkId;
          highlightEvidenceCard(chunkId);
        });
      });
    }

    // Meta Info
    if (metaInfo) {
      metaInfo.textContent = `Retrieved in ${elapsedMs}ms • ${evaluation.results.length} chunks evaluated • ${answerResult.citations.length} sources cited`;
    }
  }

  function renderEvidenceList(results, queryTokens) {
    const container = document.getElementById('evidenceList');
    const badge = document.getElementById('evidenceCountBadge');
    if (!container) return;

    if (!results || results.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <p class="empty-text">No evidence found in corpus matching the current question.</p>
        </div>
      `;
      if (badge) badge.textContent = '0 Chunks';
      return;
    }

    if (badge) badge.textContent = `${results.length} Chunks Retrieved`;

    let html = '';
    results.forEach((item, idx) => {
      const chunk = item.chunk;
      const score = item.rawScore;
      const matched = item.matchedTerms || [];

      // Highlight matching query tokens in the chunk text
      let highlightedText = escapeHtml(chunk.text);
      matched.forEach(term => {
        if (term.length > 2) {
          const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
          highlightedText = highlightedText.replace(regex, '<mark>$1</mark>');
        }
      });

      html += `
        <div class="evidence-card" data-chunk-id="${chunk.id}">
          <div class="evidence-card-header">
            <span class="chunk-id-tag">${chunk.id}</span>
            <span class="chunk-doc-title" title="${escapeHtml(chunk.docTitle)}">${escapeHtml(chunk.docTitle)}</span>
            <span class="score-badge" title="Derived lexical similarity score">BM25: ${score}</span>
          </div>
          <div class="evidence-text" id="evText-${chunk.id}">${highlightedText}</div>
          <div class="evidence-card-footer">
            <span>Matches: ${matched.length > 0 ? matched.join(', ') : 'None'}</span>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-secondary btn-sm toggle-chunk-btn" data-chunk-id="${chunk.id}" style="padding: 2px 8px; font-size: 0.72rem;">
                Expand
              </button>
              <button class="btn btn-secondary btn-sm inspect-chunk-btn" data-chunk-id="${chunk.id}" style="padding: 2px 8px; font-size: 0.72rem;">
                Full Modal
              </button>
            </div>
          </div>
        </div>
      `;
    });

    container.innerHTML = html;

    // Attach listeners to toggle expand/collapse buttons
    container.querySelectorAll('.toggle-chunk-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const chunkId = btn.dataset.chunkId;
        const textEl = document.getElementById(`evText-${chunkId}`);
        if (textEl) {
          if (textEl.classList.contains('expanded')) {
            textEl.classList.remove('expanded');
            btn.textContent = 'Expand';
          } else {
            textEl.classList.add('expanded');
            btn.textContent = 'Collapse';
          }
        }
      });
    });

    // Attach listeners to inspect buttons
    container.querySelectorAll('.inspect-chunk-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        openChunkModal(btn.dataset.chunkId);
      });
    });
  }

  function handleClear() {
    const input = document.getElementById('questionInput');
    const answerBody = document.getElementById('answerBody');
    const verdictBadge = document.getElementById('verdictBadge');
    const cragContainer = document.getElementById('cragNoticeContainer');
    const metaInfo = document.getElementById('pipelineMetaInfo');
    const evidenceList = document.getElementById('evidenceList');
    const evidenceBadge = document.getElementById('evidenceCountBadge');
    const statLatency = document.getElementById('statLatency');

    if (input) input.value = '';
    if (statLatency) statLatency.textContent = 'Ready';
    if (answerBody) {
      answerBody.innerHTML = 'Enter a question above or select an example query to execute the local retrieval pipeline and generate an evidence-grounded answer with source citations.';
    }
    if (verdictBadge) {
      verdictBadge.className = 'verdict-badge relevant';
      verdictBadge.innerHTML = '<span class="status-pulsar" style="width: 6px; height: 6px;"></span> Ready for Query';
    }
    if (cragContainer) {
      cragContainer.style.display = 'none';
      cragContainer.innerHTML = '';
    }
    if (metaInfo) {
      metaInfo.textContent = 'Status: Idle • 0 sources cited';
    }
    if (evidenceList) {
      evidenceList.innerHTML = `
        <div class="empty-state">
          <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <p class="empty-text">No question asked yet.<br>Retrieved document chunks will appear here with calculated BM25 relevance scores, highlighted tokens, and expandable full context.</p>
        </div>
      `;
    }
    if (evidenceBadge) evidenceBadge.textContent = 'Top 3 Chunks';

    updatePipelineTrack(0);
    state.currentResults = null;
  }

  // ==========================================
  // 10. UTILITY FUNCTIONS
  // ==========================================
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // ==========================================
  // 11. INITIALIZATION & BINDINGS
  // ==========================================
  document.addEventListener('DOMContentLoaded', () => {
    // 1. Load initial corpus
    loadDocumentsFromStorage();
    if (state.documents.length > 0) {
      state.selectedDocId = state.documents[0].id;
    }
    renderDocumentList();
    if (state.selectedDocId) {
      selectDocument(state.selectedDocId);
    }

    // 2. Navigation Tab Switching
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('pageTitle');
    const pageDesc = document.getElementById('pageDesc');

    const tabDescriptions = {
      'qa': {
        title: 'QA Workspace',
        desc: 'Intelligent document retrieval and evidence-grounded question answering'
      },
      'docs': {
        title: 'Document Library',
        desc: 'Inspect preloaded corpus, chunk partitions, and add custom research documents'
      },
      'methodology': {
        title: 'Methodology & Research',
        desc: 'Academic evaluation of DPR, CRAG, RAPTOR, and ColPali frameworks'
      },
      'architecture': {
        title: 'System Architecture',
        desc: 'End-to-end flowchart from query processing to evidence-grounded citation synthesis'
      },
      'guide': {
        title: 'Presentation Guide',
        desc: 'Classroom walkthrough script and key academic talking points'
      }
    };

    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const targetTab = item.dataset.tab;
        navItems.forEach(n => n.classList.remove('active'));
        item.classList.add('active');

        tabContents.forEach(tc => {
          tc.classList.remove('active');
          if (tc.id === `tab-${targetTab}`) {
            tc.classList.add('active');
          }
        });

        if (tabDescriptions[targetTab]) {
          if (pageTitle) pageTitle.textContent = tabDescriptions[targetTab].title;
          if (pageDesc) pageDesc.textContent = tabDescriptions[targetTab].desc;
        }
      });
    });

    // 3. Query Execution Buttons
    const askBtn = document.getElementById('askBtn');
    const clearBtn = document.getElementById('clearBtn');
    const questionInput = document.getElementById('questionInput');

    if (askBtn && questionInput) {
      askBtn.addEventListener('click', () => {
        handleAskQuestion(questionInput.value);
      });

      questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          handleAskQuestion(questionInput.value);
        }
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', handleClear);
    }

    // 4. Suggested Query Chips
    const chips = document.querySelectorAll('.chip-btn');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        const query = chip.dataset.query;
        if (questionInput) questionInput.value = query;
        handleAskQuestion(query);
      });
    });

    // 5. Add Document Form
    const addDocBtn = document.getElementById('addDocBtn');
    const newDocTitle = document.getElementById('newDocTitle');
    const newDocContent = document.getElementById('newDocContent');

    if (addDocBtn && newDocTitle && newDocContent) {
      addDocBtn.addEventListener('click', () => {
        const success = handleAddDocument(newDocTitle.value, newDocContent.value);
        if (success) {
          newDocTitle.value = '';
          newDocContent.value = '';
          alert('Document successfully indexed into active corpus!');
        }
      });
    }

    // 6. File Dropzone (.txt upload)
    const dropzone = document.getElementById('fileDropzone');
    const fileInput = document.getElementById('fileInput');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary)';
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--neutral-300)';
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--neutral-300)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          processUploadedFile(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          processUploadedFile(e.target.files[0]);
        }
      });
    }

    function processUploadedFile(file) {
      if (!file.name.endsWith('.txt')) {
        alert('Validation error: Only plain text (.txt) files are supported.');
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target.result;
        if (!content || !content.trim()) {
          alert('Validation error: The selected file is empty.');
          return;
        }

        const title = file.name.replace(/\.txt$/i, '').replace(/[-_]/g, ' ');
        if (newDocTitle) newDocTitle.value = title;
        if (newDocContent) newDocContent.value = content;
      };

      reader.onerror = () => {
        alert('Error reading uploaded file.');
      };

      reader.readAsText(file);
    }

    // 7. Reset Default Documents
    const resetDocsBtn = document.getElementById('resetDocsBtn');
    if (resetDocsBtn) {
      resetDocsBtn.addEventListener('click', () => {
        if (confirm('Reset document library to default preloaded educational documents?')) {
          localStorage.removeItem(STORAGE_KEY);
          loadDocumentsFromStorage();
          state.selectedDocId = state.documents[0].id;
          renderDocumentList();
          selectDocument(state.selectedDocId);
        }
      });
    }

    // 8. Modal Close Controls
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalDoneBtn = document.getElementById('modalDoneBtn');
    const chunkModal = document.getElementById('chunkModal');

    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeChunkModal);
    if (modalDoneBtn) modalDoneBtn.addEventListener('click', closeChunkModal);
    if (chunkModal) {
      chunkModal.addEventListener('click', (e) => {
        if (e.target === chunkModal) closeChunkModal();
      });
    }

    // 9. Copy Answer Functionality
    const copyBtn = document.getElementById('copyAnswerBtn');
    const copyBtnText = document.getElementById('copyBtnText');

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const answerBody = document.getElementById('answerBody');
        if (!answerBody) return;

        // Extract clean text without HTML tags and citations
        const cleanText = answerBody.innerText || answerBody.textContent || '';
        if (!cleanText.trim()) return;

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(cleanText).then(() => {
            if (copyBtnText) copyBtnText.textContent = 'Copied!';
            copyBtn.style.color = 'var(--accent-cyan)';
            setTimeout(() => {
              if (copyBtnText) copyBtnText.textContent = 'Copy';
              copyBtn.style.color = '';
            }, 2000);
          }).catch(err => {
            console.warn('Clipboard write failed:', err);
          });
        }
      });
    }
  });

  // Export internal methods to window for automated testing verification
  window.__DocuMindTestAPI = {
    DEFAULT_DOCUMENTS,
    STOP_WORDS,
    SYNONYM_MAP,
    state,
    chunkDocument,
    rebuildIndex,
    processQuery,
    retrieveEvidence,
    evaluateAndCorrectRetrieval,
    synthesizeAnswer,
    handleAddDocument,
    deleteCustomDocument
  };
  window.__RAGnitionTestAPI = window.__DocuMindTestAPI;
})();
