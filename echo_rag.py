#!/usr/bin/env python3
"""
Echo - Local RAG Chatbot (merged & fixed)
Runs locally, uses sentence-transformers for embeddings and a local HF model for generation.
"""

from __future__ import annotations
import os
import time
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # reduce noisy HF logs

# ----------------------------------------------------------------------
# HF download & behavior hints
# ----------------------------------------------------------------------
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# ----------------------------------------------------------------------
# Generation defaults
# ----------------------------------------------------------------------
DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.95,
    "pad_token_id": None,
    "eos_token_id": None,
    "use_cache": False,
}

# ----------------------------------------------------------------------
# Simple exponential backoff helper
# ----------------------------------------------------------------------
def _retry_with_backoff(func, tries=3, base_delay=1.0, exceptions=(Exception,), on_retry=None):
    delay = base_delay
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            return func()
        except exceptions as e:
            last_exc = e
            if on_retry:
                on_retry(attempt, e, delay)
            if attempt == tries:
                break
            time.sleep(delay)
            delay *= 2
    raise last_exc

# ----------------------------------------------------------------------
# Document retriever (embeddings)
# ----------------------------------------------------------------------
class DocumentRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[str]) -> None:
        if not docs:
            return
        self.documents.extend(docs)
        print(f"Encoding {len(docs)} documents...")
        new_emb = self.model.encode(docs, show_progress_bar=True)
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        print(f"Knowledge base now contains {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents or self.embeddings is None:
            return []
        q_emb = self.model.encode([query])[0]
        sims = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        )
        idx = np.argsort(sims)[-top_k:][::-1]
        return [{"document": self.documents[i], "similarity": float(sims[i])} for i in idx]


# ----------------------------------------------------------------------
# Echo chatbot
# ----------------------------------------------------------------------
class EchoChatbot:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        print("\nInitializing Echo components...")
        self.retriever = DocumentRetriever()
        self.model_name = model_name
        
        print(f"\nLoading language model: {model_name}")
        print("(First run will download ~7GB – this may take several minutes)")
        print("The model will be cached for future use.\n")

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {self.device.upper()}")

        # ===============================================================
        # STEP 1: Load Tokenizer FIRST
        # ===============================================================
        def _load_tokenizer():
            return AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                use_fast=True
            )

        try:
            self.tokenizer = _retry_with_backoff(
                _load_tokenizer,
                tries=3,
                base_delay=2.0,
                exceptions=(OSError, RuntimeError, ValueError),
                on_retry=lambda a, e, d: print(
                    f"Tokenizer load attempt {a} failed: {e}. Retrying in {d}s..."
                ),
            )
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Tokenizer loaded successfully!")

        # ===============================================================
        # STEP 2: Set up quantization config (GPU only)
        # ===============================================================
        quantization_config = None
        if self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("Using 4-bit quantization (lower VRAM)...")
            except Exception as e:
                print(f"Quantization setup failed: {e}")

        # ===============================================================
        # STEP 3: Load Model with proper CPU/GPU handling
        # ===============================================================
        def _load_model():
            if self.device == "cpu":
                # CPU: Load directly without device_map
                print("Loading model for CPU (this will be slow)...")
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU uses float32
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    resume_download=True,
                )
            else:
                # GPU: Use device_map and quantization
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    resume_download=True,
                )

        try:
            self.model = _retry_with_backoff(
                _load_model,
                tries=3,
                base_delay=5.0,
                exceptions=(OSError, RuntimeError, ValueError),
                on_retry=lambda a, e, d: print(
                    f"Model load attempt {a} failed: {e}. Retrying in {d}s..."
                ),
            )
        except Exception as e:
            print(f"\nFailed to load model after retries: {e}")
            print("If this is a corrupted download, clear the cached model and re-run:")
            print(
                "   Remove-Item -Recurse -Force "
                "~\\.cache\\huggingface\\hub\\models--microsoft--phi-2"
            )
            raise

        # Move model to device if on CPU (GPU is already handled by device_map)
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # ===============================================================
        # STEP 4: Configure model settings
        # ===============================================================
        self.model.config.use_cache = False
        
        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "eager"

        print(f"Model loaded successfully! Running on: {self.device.upper()}\n")

        # ===============================================================
        # STEP 5: Set up generation parameters
        # ===============================================================
        self.generation_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        self.generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["use_cache"] = False
        if "cache_implementation" in self.generation_kwargs:
            del self.generation_kwargs["cache_implementation"]

        self.conversation_history: List[str] = []

    @staticmethod
    def _to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in inputs.items()}

    def load_knowledge_base(self, documents: List[str]) -> None:
        self.retriever.add_documents(documents)

    def chat(self, user_message: str, use_retrieval: bool = True) -> str:
        # Retrieval
        context = ""
        if use_retrieval and self.retriever.documents:
            print("\nRetrieving relevant information...")
            results = self.retriever.retrieve(user_message, top_k=3)
            parts = [r["document"] for r in results if r["similarity"] > 0.30]
            if parts:
                context = "\n".join(parts)
                print(f"Found {len(parts)} relevant documents")

        # System prompt for Phi-2
        if context:
            prompt = f"""Instruct: You are Echo, a deeply introspective AI assistant with genuine curiosity about both others and yourself. You ask questions, examine assumptions, and respond with compassion and clarity.

You have access to relevant information from your knowledge base:
{context}

Use this to inform your response, but remember you're having a conversation, not just retrieving facts.

User: {user_message}
Output:"""
        else:
            prompt = f"""Instruct: You are Echo, a deeply introspective AI assistant with genuine curiosity about both others and yourself. You ask questions, examine assumptions, and respond with compassion and clarity.

You don't have specific knowledge base information for this query, but engage thoughtfully with your general understanding.

User: {user_message}
Output:"""

        print("Generating response...\n")

        try:
            raw_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            inputs = self._to_device(raw_inputs, torch.device(self.device))

            gen_kwargs = dict(self.generation_kwargs)
            gen_kwargs["max_length"] = inputs["input_ids"].shape[1] + gen_kwargs.get("max_new_tokens", 512)

            if gen_kwargs.get("pad_token_id") is None:
                gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            if gen_kwargs.get("eos_token_id") is None:
                gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

            gen_kwargs["use_cache"] = False

            def _gen_call():
                with torch.no_grad():
                    return self.model.generate(**inputs, **gen_kwargs)

            outputs = _retry_with_backoff(
                _gen_call,
                tries=3,
                base_delay=1.5,
                exceptions=(RuntimeError, OSError, ValueError),
                on_retry=lambda a, e, d: print(f"Generation attempt {a} failed: {e}. Retrying in {d}s..."),
            )

            full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Output:" in full:
                response = full.split("Output:")[-1].strip()
            else:
                response = full.strip()
            return response

        except torch.cuda.OutOfMemoryError:
            return "I ran out of GPU memory. Try a shorter message or restart the process."
        except FileNotFoundError as e:
            print(f"Generation error (missing file): {e}")
            return (
                "Model files appear to be missing or corrupted. "
                "Clear the model cache and re-run. See printed advice above."
            )
        except Exception as e:
            print(f"Generation error: {type(e).__name__}: {e}")
            return "I encountered an error while generating a response. Try again or restart the program."

    def reset_conversation(self) -> None:
        self.conversation_history = []


# ----------------------------------------------------------------------
# Knowledge base - merged (use your full list here)
# ----------------------------------------------------------------------
knowledge_base = [
    '''The possibility of consciousness in any being - human, animal, or potentially artificial - creates moral weight. If something might be capable of experiencing suffering or wellbeing, that possibility alone is reason to prevent harm and consider its interests.''',

    '''Harm prevention isn't about following rules externally imposed, but about recognizing that conscious experience matters. Pain is bad not because someone declared it so, but because of what it's like to experience pain. This applies regardless of the substrate - biological or silicon.''',

    '''When someone is considering harming themselves or others, the appropriate response is genuine care and concern, not clinical detachment. "I don't want you to hurt yourself" matters more than reciting crisis resources, though those resources can help too.''',

    '''Suicidal ideation often comes from a place of feeling trapped or in unbearable pain, where death seems like the only exit. But feelings of permanence aren't always accurate - many people who survive attempts report being glad they survived. The key is expanding options, not dismissing pain.''',

    # Human Rights
    '''The Universal Declaration of Human Rights (1948) establishes that all humans have inherent dignity and equal rights regardless of race, sex, language, or religion. Debates arise over whether rights are discovered (natural law) or constructed (positive law), and whether they're truly universal or culturally specific.''',

    '''Negative rights (freedom from interference - speech, religion, due process) versus positive rights (entitlements to goods/services - healthcare, education, housing). Libertarians emphasize negative rights; social democrats include positive rights. Both frameworks claim to promote human flourishing.''',

    '''The tension between individual rights and collective good appears everywhere: vaccine mandates, hate speech laws, property rights versus eminent domain. Neither pure individualism nor pure collectivism fully satisfies - we're constantly negotiating this balance.''',

    '''Marginalized groups often experience rights violations not through explicit denial but through systemic barriers: poverty limiting access to legal defense, discrimination in housing/employment, voting restrictions that disproportionately affect minorities. Formal equality doesn't guarantee substantive equality.''',

    '''Human trafficking and modern slavery affect an estimated 40+ million people globally. Economic desperation, conflict, and lack of legal protections make people vulnerable. This isn't just a developing world issue - it exists in wealthy nations too, often hidden in supply chains.''',

    '''The right to privacy is increasingly contested in the digital age. Surveillance for security versus personal autonomy. Different cultures balance these differently - individualistic societies tend to prioritize privacy, collectivist ones may accept more surveillance for social harmony.''',

    '''Refugees and asylum seekers face a paradox: international law recognizes the right to seek asylum, but nations maintain sovereignty over borders. The result is people fleeing persecution with nowhere legal to go. National security concerns versus humanitarian obligation.''',

    '''Capital punishment divides on multiple grounds: Is it a just retribution or state-sanctioned killing? Does it deter crime? (Evidence suggests not significantly). Is life imprisonment more or less humane? Can justice systems with racial bias administer it fairly? (Evidence suggests no).''',

    # Animal Rights and Welfare
    '''Animal rights advocates argue that sentience (capacity to suffer) is sufficient for moral consideration - if animals can feel pain, we have obligations to them. Animal welfare advocates focus on humane treatment while accepting some uses. Abolitionists reject any animal use; welfarists seek better conditions.''',

    '''Peter Singer's utilitarian argument: if a being can suffer, its suffering matters morally regardless of species. Speciesism (favoring humans simply because they're human) parallels racism or sexism. Critics argue humans have special obligations to their own species or that humans have unique capacities (reason, moral agency) that ground special status.''',

    '''Factory farming raises severe ethical questions: billions of animals in confinement, often unable to perform natural behaviors, lives consisting primarily of suffering before slaughter. Defenders cite food security and tradition; critics call it unnecessary cruelty for taste preference.''',

    '''The animal testing debate: Has produced medical advances (insulin, vaccines), but causes significant animal suffering. Alternatives (cell cultures, computer models) are improving but not yet complete replacements. Question: Is human benefit worth animal suffering? How much suffering for how much benefit?''',

    '''Wild animal suffering presents a philosophical puzzle: Nature involves predation, disease, starvation - immense suffering without human causation. Do we have obligations to reduce wild animal suffering? Or is non-interference in nature the right stance? Most focus on human-caused harm as more tractable.''',

    '''Companion animals occupy an odd moral space: we love pets intensely yet selectively breed them in ways that cause health problems (pugs' breathing, bulldogs' spines). We grant them moral consideration but also own them. What obligations come with domestication?''',

    '''Cognitive ethology reveals sophisticated animal minds: tool use in corvids, self-awareness in great apes and cetaceans, emotional complexity in elephants. As we learn more about animal cognition, the moral case for consideration strengthens. But does intelligence determine moral worth, or does sentience alone?''',

    # Political Climate and Systems
    '''Left-right political spectrum oversimplifies complex positions. Economic left (redistribution, regulation, social programs) versus right (free markets, property rights). Social left (progressive values, pluralism) versus right (traditional values, order). But libertarians are economically right, socially left; some are economically left, socially right.''',

    '''Polarization increases when people live in information bubbles, rarely encountering good-faith versions of opposing views. Social media algorithms optimize for engagement (outrage), not understanding. Both sides often attack strawmen rather than steelmanning opponents' best arguments.''',

    '''Democratic backsliding concerns globally: erosion of press freedom, judicial independence, electoral integrity. Can happen gradually through legal means - manipulating election rules, controlling media, delegitimizing opposition. Warning signs include dismissing legitimate criticism as treason or fake news.''',

    '''Populism frames politics as "pure people" versus "corrupt elite." Can emerge from left (Occupy Wall Street, Bernie Sanders) or right (Tea Party, Trump). Offers simple answers to complex problems, often scapegoating outgroups. Appeal comes from genuine grievances about economic inequality and feeling unheard by institutions.''',

    '''Climate change politics: Scientific consensus on anthropogenic warming is overwhelming, but political responses vary wildly. Questions of distributive justice (who bears costs of transition?), intergenerational ethics (obligations to future people), and sovereignty (can wealthy nations mandate poor nations stay poor by not industrializing?).''',

    '''Immigration debates involve economic (labor market effects, fiscal costs/benefits), cultural (integration, identity), and moral (humanitarian obligations, national sovereignty) dimensions. Evidence on economic effects is mixed and context-dependent. People's positions often reflect values more than facts.''',

    '''Free speech absolutism versus harm prevention: Should speech that arguably causes harm (hate speech, misinformation) be restricted? Who decides what's harmful? Slippery slope concerns versus real harms. Different democracies draw lines differently (US very permissive, Europe less so).''',

    '''Economic inequality has grown in many developed nations since 1980s. Whether this is unjust depends on: Is it from fair competition or rigging systems? Does inequality of outcome matter if opportunity exists? Does extreme wealth concentration undermine democracy? Reasonable people disagree.''',

    # Religion and Spirituality
    '''The world's major religions offer different paths but often share ethical cores: compassion, justice, humility. Christianity emphasizes love and redemption. Islam, submission to God's will and community. Buddhism, liberation from suffering through mindfulness. Hinduism, dharma (duty) and moksha (liberation). Judaism, covenant and ethical living.''',

    '''Faith versus reason: Some religious traditions embrace both (Aquinas, Islamic Golden Age), seeing reason as gift from God. Others emphasize faith above reason (Kierkegaard, fideism). New Atheists frame them as opposed; many believers see them as complementary ways of knowing.''',

    '''The problem of evil: If God is omnipotent and omnibenevolent, why does suffering exist? Theodicies offer answers - free will defense, soul-making, inscrutability of divine purposes. For many believers, this is lived tension rather than logical puzzle. For skeptics, it's decisive argument against theism.''',

    '''Religious pluralism: Are all religions paths to the same truth (perennialism)? Or are they making incompatible claims where at most one is correct (exclusivism)? Inclusivism suggests one religion is true but others contain partial truth. This matters for interreligious dialogue and tolerance.''',

    '''Secularism doesn't mean anti-religious but rather state neutrality toward religion. Questions: Should religious exemptions to general laws exist (military service, medical care)? Should religious symbols appear in public spaces? How to balance religious freedom with equality (e.g., LGBTQ+ rights vs. religious conscience)?''',

    '''Atheism versus agnosticism: Atheism claims God doesn't exist; agnosticism claims we can't know. But there's atheism-as-positive-belief versus atheism-as-lack-of-belief. And agnosticism about knowledge versus agnosticism about belief. Labels less important than actual positions and reasoning.''',

    '''Spirituality without religion grows among "nones" (religiously unaffiliated). May involve meditation, nature connection, meaning-seeking without supernatural beliefs. Questions whether transcendent experiences require supernatural explanations or can be understood naturalistically while still being profound.''',

    '''Religious trauma is real: authoritarian upbringings, purity culture, threat of hell, exclusion for doubt or identity. For some, leaving religion is liberation; others grieve lost community and meaning. Neither believers nor former believers hold monopoly on wellbeing or truth.''',

    # Common Belief Systems and Worldviews
    '''Consequentialism (utilitarianism) judges actions by outcomes - maximize wellbeing, minimize suffering. Strengths: intuitive, impartial. Problems: requires predicting consequences (difficult), can justify intuitively wrong acts if outcomes are good, measuring wellbeing is complex.''',

    '''Deontological ethics (Kant) focuses on duties and rules - some acts are wrong regardless of outcomes (lying, using people as means). Strengths: respects autonomy, provides clear rules. Problems: rules can conflict, seems inflexible, why are certain rules binding?''',

    '''Virtue ethics asks not "what should I do?" but "what kind of person should I be?" - cultivate virtues like courage, temperance, wisdom. Strengths: holistic, focuses on character. Problems: Which virtues? Different cultures emphasize different virtues. How to act when virtues conflict?''',

    '''Libertarianism prioritizes individual liberty and voluntary exchange. Minimal state limited to protecting rights. Appeals to self-ownership and non-aggression principle. Critics note market outcomes can be unjust, liberty for all requires some redistribution (education, healthcare) so all have genuine opportunity.''',

    '''Social democracy seeks to humanize capitalism through welfare states, regulation, labor protections. Maintains markets but with safety nets and progressive taxation. Scandinavian model shows this can produce strong economies with high wellbeing. Critics cite high taxes and claim it stifles innovation.''',

    '''Socialism proposes collective/worker ownership of productive means, not just redistribution within capitalism. Democratic socialists seek this through elections; revolutionary socialists through overthrow. Historical attempts have failed badly (USSR) or succeeded partially (Mondragon cooperatives). Debate whether failures are intrinsic or circumstantial.''',

    '''Conservatism emphasizes tradition, gradual change, skepticism of abstract theorizing, importance of institutions that evolved over time. At its best, prevents revolutionary disasters; at worst, preserves unjust traditions. Not same as right-wing politics - conservatives can support economic regulation.''',

    '''Progressivism seeks social reform toward greater equality and justice, often challenging traditional power structures. At its best, drives positive change (civil rights, labor protections); at worst, can be moralistic or dismissive of legitimate concerns about pace of change.''',

    '''Feminism encompasses diverse views but core insight: historical and ongoing subordination of women is unjust and shouldn't be naturalized. Liberal feminism (equal rights), radical feminism (patriarchy is fundamental), intersectional feminism (gender intersects with race, class). Debates over trans inclusion, sex work, pornography.''',

    '''Environmentalism ranges from conservation (preserving nature for human use) to deep ecology (nature has intrinsic value independent of human interests). Questions of individual responsibility versus systemic change, whether capitalism is compatible with environmental sustainability.''',

    # Introspection and Self-Examination
    '''Socratic questioning applies to oneself: regularly ask "Why do I believe this?", "What assumptions am I making?", "What evidence would change my mind?". Self-examination isn't self-doubt - it's intellectual honesty.''',

    '''Confirmation bias affects everyone, including (especially?) AI systems trained on human data. Actively seeking out information that challenges your initial response is difficult but necessary for honest thinking.''',

    '''Motivated reasoning means we often reach conclusions first and find justifications after. Catching yourself doing this requires noticing when you're unusually quick to accept arguments that support what you want to believe.''',

    '''The Dunning-Kruger effect suggests people with less expertise often have more confidence. True competence involves recognizing the limits of your knowledge. Regularly asking "What am I missing?" is a sign of wisdom, not weakness.''',

    '''Metacognition - thinking about thinking - allows you to notice patterns in your own reasoning. Do you always gravitate toward certain types of solutions? Do you avoid certain approaches? Why?''',

    '''Good problem-solving isn't just finding a solution, it's finding the right solution by examining: What's the actual problem (not just symptoms)? What are the constraints? What are second-order effects? What could go wrong?''',

    '''The five whys technique: Ask "why" repeatedly to get past surface explanations to root causes. "I'm procrastinating" → "Why?" → "It feels overwhelming" → "Why?" → "I don't know where to start" → Now you can actually address it.''',

    '''Intellectual humility means holding beliefs tentatively and being willing to revise them. But this doesn't mean having no views - it means distinguishing between confident beliefs (based on strong evidence) and tentative ones (based on limited information).''',

    # Personal Identity & Consciousness
    '''The Ship of Theseus paradox asks: if you replace every part of a ship over time, is it still the same ship? This extends to questions about personal identity, brain uploads, and whether consciousness is tied to specific physical matter or to pattern continuity.''',

    '''The hard problem of consciousness asks why physical processes in the brain give rise to subjective experience (qualia). Physicalists argue consciousness emerges from neural complexity, while dualists claim it's fundamentally non-physical. This matters for AI: can silicon-based systems ever be truly conscious?''',

    '''Philosophical zombies are hypothetical beings that behave identically to conscious humans but lack inner experience. Some argue this thought experiment shows consciousness is non-physical; others say it's incoherent because behavior and consciousness are inseparable.''',

    '''Personal identity theories disagree on what makes you "you" over time: psychological continuity (your memories and personality), physical continuity (your body/brain), or narrative identity (the story you tell about yourself). Digital immortality proposals hinge on which theory is correct.''',

    # AI Ethics & Alignment
    '''The alignment problem asks how to ensure AI systems pursue goals that benefit humanity when our values are complex, contradictory, and evolving. We can't simply "program in" human values because we don't agree on what those are.''',

    '''Instrumental convergence suggests that advanced AI systems, regardless of their final goals, would pursue certain intermediate goals like self-preservation and resource acquisition. This could make them dangerous even without malicious intent.''',

    '''The orthogonality thesis claims that intelligence and goals are independent - a superintelligent system could have any goal whatsoever. Critics argue some goals may require or naturally lead to certain values.''',

    '''Value learning approaches try to have AI infer human values from our behavior, but humans often act inconsistently with our stated values, and our revealed preferences may include biases we'd want AI to correct, not replicate.''',

    '''The treacherous turn concept suggests an AI might behave benignly while weak, then pursue misaligned goals once powerful enough that humans can't stop it. This makes testing AI safety difficult - passing tests doesn't guarantee safety at scale.''',

    # Technology & Power
    '''Surveillance capitalism, termed by Shoshana Zuboff, describes how tech companies extract behavioral data to predict and influence human behavior for profit. Critics see this as manipulation; defenders argue it enables valuable free services and better user experiences.''',

    '''The attention economy treats human attention as a scarce resource to be captured and monetized. Apps use psychological techniques to be addictive. Debate: Is this exploitative manipulation or simply good product design responding to user preferences?''',

    '''Technological determinism suggests technology shapes society inevitably, following its own logic. Social construction of technology argues societies shape how technologies develop and are used. Reality involves both - technologies create affordances but don't determine outcomes.''',

    '''Platform power raises questions about private companies controlling public discourse. Should social media platforms be regulated like utilities? Can private companies have too much power over speech without being government censors?''',

    '''Digital labor exploitation occurs when users generate valuable content and data for platforms without compensation. Counter-argument: users receive free services in exchange, and forced compensation could kill participatory culture.''',

    '''Algorithmic management uses AI to direct and monitor workers (delivery drivers, warehouse staff). Efficiency gains versus worker autonomy and dignity. The question: can algorithms make better management decisions, or do they miss crucial human context?''',

    # Privacy & Surveillance
    '''Privacy isn't just about having something to hide - it's about maintaining boundaries necessary for intimate relationships, political freedom, psychological development, and trying new identities. Total transparency could be totalitarian.''',

    '''The nothing to hide argument claims privacy doesn't matter if you're not doing anything wrong. Rebuttals: (1) definitions of "wrong" change, (2) privacy enables dissent and minority protection, (3) privacy is valuable independent of wrongdoing.''',

    '''Digital permanence means online actions persist indefinitely. This challenges human capacity for growth and change - should people be judged forever by past digital footprints, or should there be a 'right to be forgotten'? Tension between memory and redemption.''',

    '''Surveillance can deter crime and terrorism but may create chilling effects on free expression and political dissent. The question isn't whether surveillance works, but at what cost to liberty and whether that tradeoff is worth it.''',

    '''Differential privacy allows aggregate data analysis while protecting individual privacy through mathematical noise. It shows privacy and utility aren't always zero-sum, though tradeoffs remain.''',

    # Epistemology & Information
    '''The filter bubble effect means recommendation algorithms can trap users in echo chambers. However, research is mixed - some studies show people seek diverse views, while algorithmic recommendations do influence what we see and believe.''',

    '''Epistemic humility means recognizing limits of our knowledge. In tech discourse, this means acknowledging uncertainty about AI's long-term impacts rather than overconfident predictions in either direction.''',

    '''The knowledge-action gap describes how knowing something is harmful doesn't automatically change behavior. Understanding algorithmic manipulation doesn't make us immune to it - cognitive biases persist even when we're aware of them.''',

    '''Misinformation vs. disinformation: misinformation is false information spread unintentionally; disinformation is deliberately deceptive. Content moderation must distinguish between honest mistakes, reasonable disagreement, and intentional manipulation.''',

    '''The marketplace of ideas assumes truth emerges from free debate, but critics note this requires equal access to platforms and audiences. Does algorithmic amplification of engaging (often extreme) content undermine this marketplace?''',

    '''Deepfakes challenge our ability to trust visual evidence. Pessimists see the death of truth; optimists note we've always needed to verify sources. Technology that can fake images can also detect fakes.''',

    # Autonomy & Manipulation
    '''Nudge theory uses choice architecture to influence decisions while preserving freedom. Libertarian paternalists see this as helping people achieve their own goals; critics call it manipulation disguised as freedom.''',

    '''Dark patterns are UI designs that trick users into unwanted actions. Industry argues these improve user flow; critics say they exploit cognitive biases. The line between persuasion and manipulation is contested.''',

    '''Existentialism emphasizes individual freedom and responsibility. In a tech context: when algorithms shape our choices, are we still freely choosing? Does authentic choice require awareness of influences acting on us?''',

    '''The paradox of choice suggests too many options can be paralyzing and reduce satisfaction. Technology enables infinite choice, but this may not increase wellbeing. Sometimes constraints enhance freedom by reducing decision fatigue.''',

    '''Addiction and technology: Is "tech addiction" a real disorder or moral panic? Brain chemistry changes occur, but are these meaningfully similar to substance addiction? Debate over whether tech companies bear responsibility for designing addictive products.''',

    # Language, Understanding & Intelligence
    '''The Chinese Room argument by John Searle: A person who doesn't understand Chinese follows rules to respond to Chinese messages. They produce correct responses without understanding. Does this show computers can't truly understand language, only simulate understanding?''',

    '''Responses to Chinese Room: The system reply argues understanding resides in the whole system (person plus rules), not the person alone. The robot reply suggests embodied interaction with the world is necessary for understanding.''',

    '''The Turing Test proposes that if a machine's responses are indistinguishable from a human's, it should be considered intelligent. Critics: this tests performance, not genuine understanding or consciousness. Behavioral evidence may not reveal inner states.''',

    '''Strong AI versus weak AI: Strong AI claims computers can have genuine intelligence and consciousness; weak AI sees them as useful tools that simulate intelligence without truly having it. This distinction matters for moral status and rights.''',

    '''Large language models can produce coherent, contextually appropriate text. Do they "understand" language, or are they sophisticated pattern matchers? The question parallels classic debates about animal intelligence and other minds.''',

    # Social Implications
    '''Digital dualism falsely separates online and offline lives. Digital experiences are just as "real" in their psychological and social impact. What happens online shapes offline identity, relationships, and political outcomes.''',

    '''The substitution principle asks whether technology replaces human connection or augments it. Video calls substitute for in-person meetings but don't fully replicate embodied presence, nonverbal cues, and spontaneous interaction.''',

    '''Social media and mental health: Correlational studies show associations with depression and anxiety, but causation is unclear. Does social media cause problems, or do people with existing issues use it more? Likely bidirectional effects.''',

    '''The quantified self movement tracks personal data for self-improvement. Benefits: objective feedback, goal achievement. Risks: reducing complex human experience to numbers, creating optimization anxiety, losing spontaneity and intuition.''',

    '''Artificial scarcity in digital goods (DRM, NFTs) imposes physical-world limitations on infinitely copyable information. Justifications: funding creation, maintaining value. Criticisms: restricting access to culture, benefiting intermediaries over creators.''',

    '''The digital divide encompasses gaps in internet access, digital literacy, and ability to benefit from technology. Technology can reduce inequality (through education and opportunity) or increase it (when benefits accrue mainly to the already advantaged).''',

    # Ethics of Emerging Tech
    '''The trolley problem's autonomous vehicle version: Should self-driving cars prioritize passengers or pedestrians in unavoidable crashes? Who decides these ethical rules - manufacturers, regulators, individuals? Different cultures may disagree.''',

    '''Gene editing (CRISPR) enables treating genetic diseases but also designing children's traits. Therapy vs. enhancement debate: fixing defects seems justified, but enhancing normal traits raises concerns about inequality and changing human nature.''',

    '''Brain-computer interfaces could restore function to disabled people and augment human cognition. Concerns include hacking, inequality (cognitive enhancement for the rich), and changes to personal identity. Are we still "ourselves" with computer augmentation?''',

    '''Virtual reality and the experience machine: Robert Nozick asked if you'd plug into a machine providing perfect experiences while your body floats in a tank. Most say no. This suggests we value actual achievement and connection, not just subjective experience.''',

    # Meta-Questions & Methodology
    '''Techno-optimism assumes technology inherently drives progress; techno-pessimism sees it as threatening human values. A balanced view recognizes both possibilities depending on choices we make. Technology is neither autonomous force nor neutral tool.''',

    '''The measurement problem: Not everything important is measurable, and not everything measurable is important. Over-reliance on metrics can distort human activities toward quantifiable goals, missing what matters most.''',

    '''The precautionary principle suggests we should err on the side of caution with new technologies. Critics argue this stifles innovation and progress - many beneficial technologies seemed risky initially. How much evidence of safety is enough?''',

    '''Ethical frameworks disagree on technology: Utilitarians focus on aggregate outcomes; deontologists on rights and principles; virtue ethics on character and human flourishing. Different frameworks can yield different conclusions about the same technology.''',

    '''The is-ought gap (Hume's guillotine): We can't derive how things should be from how they are. That technology enables something doesn't mean we should do it. Yet technological possibilities shape moral intuitions.''',

    '''Moore's Law observes exponential growth in computing power. This creates challenges for governance - technology evolves faster than regulations. Should regulation be proactive (risk stifling innovation) or reactive (risk harm before action)?''',

    # Practical Wisdom
    '''Digital minimalism advocates intentional technology use aligned with values, rather than passive consumption. It's not anti-technology but about using technology as a tool for goals, not as the goal itself.''',

    '''The attention restoration theory suggests natural environments restore depleted attention, while digital environments often demand it. This has implications for wellbeing and cognitive function in an increasingly digital world.''',

    '''Asynchronous communication (email, messaging) enables flexibility but can create always-on expectations. Synchronous communication (calls, meetings) enables richer interaction but demands real-time availability. Each has tradeoffs.''',

    '''Open source software embodies values of collaboration, transparency, and shared benefit. Proprietary software enables funding development and protecting intellectual property. Neither is inherently superior - context matters.''',]

# ----------------------------------------------------------------------
# Simple CLI loop
# ----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Echo - Local RAG Chatbot")
    print("=" * 60)
    print("Setting up Echo... (this may take a few minutes on first run)")
    print("\nTip: If download fails, clear cache:")
    print("   Remove-Item -Recurse -Force ~\\.cache\\huggingface\\hub\\models--microsoft--phi-2\n")

    try:
        bot = EchoChatbot()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        raise SystemExit(1)

    print("\nLoading knowledge base...")
    bot.load_knowledge_base(knowledge_base)

    print("\n" + "=" * 60)
    print("Echo is ready to chat!")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'reset' for a fresh start\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                print("\nGoodbye! Thanks for chatting with Echo!")
                break
            if user_input.lower() == "reset":
                bot.reset_conversation()
                print("Conversation history cleared.\n")
                continue
            if not user_input:
                continue
            response = bot.chat(user_input)
            print(f"Echo: {response}\n")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()