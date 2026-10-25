// Automated Test Suite for DocuMind AI RAG Pipeline
const fs = require('fs');
const path = require('path');

console.log('====================================================');
console.log('🧪 RUNNING RAGNITION AI COMPREHENSIVE TEST SUITE');
console.log('====================================================\n');

// 1. Verify existence of required files
const requiredFiles = ['index.html', 'style.css', 'script.js'];
let filesOk = true;
requiredFiles.forEach(f => {
  const exists = fs.existsSync(path.join(__dirname, f));
  console.log(`[File Check] ${f}: ${exists ? 'EXISTS (OK)' : 'MISSING (FAIL)'}`);
  if (!exists) filesOk = false;
});

if (!filesOk) {
  console.error('Critical failure: Missing required files.');
  process.exit(1);
}

// 2. Load and parse script.js in a mock browser environment
const scriptContent = fs.readFileSync(path.join(__dirname, 'script.js'), 'utf-8');

// Mock browser objects
const mockWindow = {};
const createMockElement = () => ({
  classList: { add: () => {}, remove: () => {} },
  addEventListener: () => {},
  appendChild: () => {},
  dataset: {},
  style: {},
  innerHTML: '',
  textContent: ''
});

const mockDocument = {
  addEventListener: () => {},
  getElementById: (id) => createMockElement(),
  createElement: (tag) => createMockElement(),
  querySelectorAll: () => []
};

// Evaluate script inside mock sandbox
const vm = require('vm');
const context = vm.createContext({
  window: mockWindow,
  document: mockDocument,
  localStorage: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {}
  },
  console: console,
  performance: { now: () => Date.now() },
  alert: (msg) => console.log('  [Alert]:', msg)
});

vm.runInContext(scriptContent, context);

const api = mockWindow.__DocuMindTestAPI;
if (!api) {
  console.error('Test API export not found on window object.');
  process.exit(1);
}

console.log('\n--- VERIFYING PHASE 3: SAMPLE DOCUMENTS & CHUNKING ---');
const docs = api.DEFAULT_DOCUMENTS;
console.log(`Total sample documents: ${docs.length}`);
if (docs.length !== 3) {
  console.error(`Expected 3 sample documents, got ${docs.length}`);
  process.exit(1);
}

const titles = docs.map(d => d.title);
console.log('Titles:', titles);
if (!titles.some(t => t.includes('Retrieval-Augmented Generation'))) {
  throw new Error('Missing RAG document');
}
if (!titles.some(t => t.includes('Artificial Intelligence and Machine Learning'))) {
  throw new Error('Missing AI/ML document');
}
if (!titles.some(t => t.includes('Climate Change'))) {
  throw new Error('Missing Climate Change document');
}

// Populate state
api.state.documents = [...docs];
api.rebuildIndex();
console.log(`Total indexed chunks across corpus: ${api.state.chunks.length}`);
if (api.state.chunks.length < 9) {
  throw new Error(`Expected at least 9 chunks across 3 documents, got ${api.state.chunks.length}`);
}

console.log('\n--- VERIFYING PHASE 4: QUERY PROCESSING & LEXICAL RETRIEVAL ---');
// Test Query 1: RAG
const q1 = "What is Retrieval-Augmented Generation?";
const tokens1 = api.processQuery(q1);
console.log(`Query 1: "${q1}"`);
console.log('Processed Tokens:', tokens1);
if (tokens1.includes('what') || tokens1.includes('is')) {
  throw new Error('Stop words were not properly filtered');
}

const retrieval1 = api.retrieveEvidence(tokens1);
console.log(`Retrieved ${retrieval1.length} passages.`);
console.log(`Top passage doc: "${retrieval1[0].chunk.docTitle}" | Chunk ID: ${retrieval1[0].chunk.id} | BM25 Score: ${retrieval1[0].rawScore}`);

if (!retrieval1[0].chunk.docTitle.includes('Retrieval-Augmented Generation')) {
  throw new Error('Query 1 did not retrieve the RAG document as top passage!');
}

// Test Query 2: RAG Benefits
const q1b = "How does document retrieval improve question answering?";
const tokens1b = api.processQuery(q1b);
console.log(`\nQuery 1b: "${q1b}"`);
console.log('Processed Tokens:', tokens1b);
const retrieval1b = api.retrieveEvidence(tokens1b);
console.log(`Top passage doc: "${retrieval1b[0].chunk.docTitle}" | Chunk ID: ${retrieval1b[0].chunk.id} | BM25 Score: ${retrieval1b[0].rawScore}`);
if (!retrieval1b[0].chunk.docTitle.includes('Retrieval-Augmented Generation')) {
  throw new Error('Query 1b did not retrieve RAG document!');
}

// Test Query 2: Climate Change
const q2 = "What causes climate change?";
const tokens2 = api.processQuery(q2);
console.log(`\nQuery 2: "${q2}"`);
console.log('Processed Tokens:', tokens2);
const retrieval2 = api.retrieveEvidence(tokens2);
console.log(`Top passage doc: "${retrieval2[0].chunk.docTitle}" | Chunk ID: ${retrieval2[0].chunk.id} | BM25 Score: ${retrieval2[0].rawScore}`);

if (!retrieval2[0].chunk.docTitle.includes('Climate Change')) {
  throw new Error('Query 2 did not retrieve Climate Change document as top passage!');
}

console.log('\n--- VERIFYING PHASE 5: CRAG CORRECTIVE RETRIEVAL & INSUFFICIENT EVIDENCE ---');
// Test Query 3: Out-of-domain query
const q3 = "What are the rules and regulations of lunar orbital spaceflight in 2050?";
const tokens3 = api.processQuery(q3);
console.log(`Query 3 (Out of domain): "${q3}"`);
console.log('Processed Tokens:', tokens3);

const initialRetrieval3 = api.retrieveEvidence(tokens3);
const cragEval3 = api.evaluateAndCorrectRetrieval(q3, tokens3, initialRetrieval3);
console.log('CRAG Verdict:', cragEval3.verdict);
console.log('CRAG Triggered:', cragEval3.cragTriggered);
console.log('CRAG Message:', cragEval3.message);

if (cragEval3.verdict !== 'insufficient') {
  throw new Error(`Expected verdict 'insufficient' for out-of-domain query, got '${cragEval3.verdict}'`);
}

const answer3 = api.synthesizeAnswer(cragEval3.effectiveTokens, cragEval3);
console.log('Answer 3 Insufficient Evidence refusal generated:');
console.log(answer3.answerHtml.replace(/<[^>]+>/g, '').trim());

if (!answer3.answerHtml.includes('Insufficient Evidence')) {
  throw new Error('System did not decline to answer out-of-domain query!');
}

console.log('\n--- VERIFYING PHASE 4: EVIDENCE GROUNDED SYNTHESIS & CITATIONS ---');
// Test synthesis for Query 1
const eval1 = api.evaluateAndCorrectRetrieval(q1, tokens1, retrieval1);
const answer1 = api.synthesizeAnswer(eval1.effectiveTokens, eval1);
console.log('Answer 1 Generated:');
console.log(answer1.answerHtml.replace(/<[^>]+>/g, '').trim());
console.log('Citations attached:', answer1.citations);

if (answer1.citations.length === 0) {
  throw new Error('Expected at least 1 citation for Query 1');
}
if (!answer1.answerHtml.includes('data-chunk-id')) {
  throw new Error('Answer HTML missing interactive citation tags');
}

console.log('\n--- VERIFYING PHASE 3: ADDING & REMOVING CUSTOM DOCUMENTS ---');
const customDocAdded = api.handleAddDocument(
  'Quantum Computing Fundamentals',
  'Quantum computing leverages the principles of quantum mechanics, specifically superposition and entanglement, to perform computations exponentially faster than classical computers for specific problem classes. Qubits can exist in a linear combination of states |0> and |1> simultaneously.'
);
console.log('Custom document addition returned:', customDocAdded);
if (!customDocAdded) throw new Error('Failed to add custom document');

const customQ = 'How do qubits differ from classical bits?';
const customTokens = api.processQuery(customQ);
const customRetrieval = api.retrieveEvidence(customTokens);
console.log(`Custom Query: "${customQ}"`);
console.log(`Top retrieved document: "${customRetrieval[0].chunk.docTitle}" | Score: ${customRetrieval[0].rawScore}`);

if (!customRetrieval[0].chunk.docTitle.includes('Quantum Computing')) {
  throw new Error('Custom document was not successfully queried!');
}

// Delete custom document
const customDocId = api.state.documents.find(d => !d.isDefault).id;
api.deleteCustomDocument(customDocId);
console.log(`Deleted custom document ${customDocId}. Remaining docs count: ${api.state.documents.length}`);
if (api.state.documents.length !== 3) {
  throw new Error('Document deletion failed');
}

console.log('\n====================================================');
console.log('✅ ALL TESTS PASSED SUCCESSFULLY!');
console.log('====================================================\n');
