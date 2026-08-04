import json
import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import threading
from intent import predict_intent, generate_chitchat_reply

PARA_MODEL = "Vamsi/T5_Paraphrase_Paws"
GRAM_MODEL = "vennify/t5-base-grammar-correction"
STYLE_MODEL = "facebook/bart-large-cnn"

with open("datasets/data.json", encoding="utf-8") as f:
    ALL_GAMES = json.load(f)

def filter_by_genre(games, genre_query):
    genre_query_lower = genre_query.lower()
    return [
        game for game in games
        if any(genre_query_lower in tag.lower() for tag in game.get("tags", []))
    ]

def filter_by_budget(games, max_price):
    return [
        game for game in games
        if game.get("price_final", float("inf")) <= max_price
    ]

def filter_by_rating(games, min_rating="any", min_ratio="any"):
    if min_rating == "any" and min_ratio == "any":
        return games

    filtered = []
    for game in games:
        rating_ok = True
        ratio_ok = True

        if min_rating != "any":
            rating_ok = min_rating.lower() in game.get("rating", "").lower()

        if min_ratio != "any":
            try:
                ratio_ok = game.get("positive_ratio", 0) >= int(min_ratio)
            except (ValueError, TypeError):
                ratio_ok = False

        if rating_ok and ratio_ok:
            filtered.append(game)

    return filtered

def filter_by_platform(games, platform_query):
    platform_query_lower = platform_query.lower()
    return [
        game for game in games
        if any(platform_query_lower in platform.lower() for platform in game.get("platforms", []))
    ]

class TextProcessor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TextProcessor, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True
        self._models_loaded = False
        self._load_event = threading.Event()

        self._load_models()

    def _load_models(self):
        print(f"Loading models on device: {self.device}")

        self.para_tok = AutoTokenizer.from_pretrained(PARA_MODEL)
        self.para_model = AutoModelForSeq2SeqLM.from_pretrained(PARA_MODEL).to(self.device).eval()
        print("Paraphraser model device:", next(self.para_model.parameters()).device)

        self.gram_tok = AutoTokenizer.from_pretrained(GRAM_MODEL)
        self.gram_model = AutoModelForSeq2SeqLM.from_pretrained(GRAM_MODEL).to(self.device).eval()
        print("Grammar model device:", next(self.gram_model.parameters()).device)

        self.style_tok = AutoTokenizer.from_pretrained(STYLE_MODEL)
        self.style_model = AutoModelForSeq2SeqLM.from_pretrained(STYLE_MODEL).to(self.device).eval()
        print("Style model device:", next(self.style_model.parameters()).device)

        self._models_loaded = True
        self._load_event.set()

    def wait_for_models(self, timeout=None):
        return self._load_event.wait(timeout=timeout)

    def paraphrase(self, text, n=5):
        inp = self.para_tok.encode("paraphrase: " + text + " </s>", return_tensors="pt", truncation=True).to(self.device)
        outs = self.para_model.generate(inp, max_length=128, num_beams=5,
                                      num_return_sequences=n, temperature=2.4,
                                      top_k=50, top_p=0.92, do_sample=True, early_stopping=True)
        cands = [self.para_tok.decode(o, skip_special_tokens=True).strip() for o in outs]
        for c in cands:
            if c.lower() != text.lower():
                return c
        return text

    def correct_grammar(self, text):
        inp = self.gram_tok.encode("fix: " + text, return_tensors="pt", truncation=True).to(self.device)
        out = self.gram_model.generate(inp, max_length=128, num_beams=4, temperature=1.0,
                                     top_k=50, top_p=0.95, do_sample=False, early_stopping=True)
        return self.gram_tok.decode(out[0], skip_special_tokens=True).strip()

    def apply_style_clean(self, text, instruction):
        prompt = f"{instruction}: {text}"
        inp = self.style_tok.encode(prompt, return_tensors="pt", truncation=True).to(self.device)
        out = self.style_model.generate(inp, max_length=128, do_sample=True,
                                      top_k=50, top_p=0.95, temperature=1.2, early_stopping=True)
        result = self.style_tok.decode(out[0], skip_special_tokens=True).strip()
        result = re.sub(rf"^{re.escape(instruction)}:\s*", "", result, flags=re.IGNORECASE)
        result = re.sub(r"(convert to active voice:|make it more exciting:)", "", result, flags=re.IGNORECASE)
        return result.strip()

    def process_description(self, desc, title, game_metadata):
        if desc.lower().startswith(title.lower()):
            desc_wo_title = desc[len(title):].lstrip(":—-. ").strip()
        else:
            desc_wo_title = desc

        parts = []
        if game_metadata.get("rating") and game_metadata.get("positive_ratio") and game_metadata.get("user_reviews"):
            parts.append(f"Rated {game_metadata['rating']} with {game_metadata['positive_ratio']}% positive reviews from {game_metadata['user_reviews']} players")
        if game_metadata.get("price_final"):
            parts.append(f"priced at just ${game_metadata['price_final']:.2f}")
            
        platforms = ", ".join(p for p in [
            "Windows" if game_metadata.get("win") else None,
            "Mac" if game_metadata.get("mac") else None,
            "Linux" if game_metadata.get("linux") else None,
            "Steam Deck" if game_metadata.get("steam_deck") else None
        ] if p)
        
        if platforms:
            parts.append(f"available on platforms like {platforms}")
            
        meta_tail = ". ".join(parts) + "." if parts else ""
        enriched_desc = f"{desc_wo_title} {meta_tail}".strip()

        paraphrased = self.paraphrase(enriched_desc)
        grammar1 = self.correct_grammar(paraphrased)
        styled1 = self.apply_style_clean(grammar1, "convert to active voice")
        grammar2 = self.correct_grammar(styled1)
        styled2 = self.apply_style_clean(grammar2, "convert to active voice")
        exciting = self.apply_style_clean(styled2, "convert to active voice")

        text = re.sub(r'([.!?])([^\s])', r'\1 \2', exciting)
        sentences = re.split('([.!?])', text)
        capitalized = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            capitalized.append(sentence + punctuation)
            
        final_text = f"{title} — {' '.join(capitalized).strip()}"
        
        final_text = re.sub(r"[\"']{2,}", "", final_text) 
        final_text = re.sub(r"\s+[\"']{1}(\s|$)", " ", final_text)
        final_text = re.sub(r"^[\"']+|[\"']+$", "", final_text)
        
        return final_text.strip()

def get_game_by_appid(app_id):
    with open("datasets/data.json", encoding="utf-8") as f:
        games = json.load(f)
    
    for game in games:
        if str(game.get('app_id')) == str(app_id):
            return game
    return None

def get_game_recommendation(app_id=None):
    processor = TextProcessor()
    
    with open("datasets/data.json", encoding="utf-8") as f:
        games = json.load(f)
    
    if app_id is not None:
        game = get_game_by_appid(app_id)
        if game is None:
            return None
    else:
        game = random.choice(games)
    
    processed_desc = processor.process_description(
        game['description'],
        game['title'],
        game
    )
    
    return {
        **game, 
        "processed_description": processed_desc  
    }


def find_top_matches(preferences, top_k=5):
    with open("datasets/data.json", encoding="utf-8") as f:
        games = json.load(f)

    def match_score(game):
        score = 0

        for tag in preferences.get("tags", []):
            if tag.lower() in map(str.lower, game["tags"]):
                score += 2

        for platform in preferences.get("platforms", []):
            if game.get(platform, False):
                score += 1

        if "max_price" in preferences and game["price_final"] <= preferences["max_price"]:
            score += 1

        if "min_rating" in preferences and game["positive_ratio"] >= preferences["min_rating"]:
            score += 1

        return score

    ranked = sorted(games, key=match_score, reverse=True)
    return ranked[:top_k]

def reset_survey():
    return {
        "genre": None,
        "budget": None,
        "rating_preference": None,
        "filtered_games": [],
        "current_index": 0,
        "confirmation": None
    }

survey_session = reset_survey()
SWITCH_INTENTS = True

def get_reply(user_msg, game_data):
    global SWITCH_INTENTS, survey_session

    intent = predict_intent(user_msg)
    print(f"Detected intent: {intent}, SWITCH_INTENTS: {SWITCH_INTENTS}")

    if SWITCH_INTENTS and intent == "recommendation":
        SWITCH_INTENTS = False

    if not SWITCH_INTENTS:
        try:
            print("Handling recommendation mode...")

            # Step 1: Begin survey
            if survey_session["genre"] is None and survey_session["budget"] is None:
                survey_session = {
                    "genre": "pending",
                    "budget": None,
                    "rating_preference": None,
                    "filtered_games": [],
                    "current_index": 0,
                    "confirmation": None
                }
                return {
                    "response": "What kind of genre are you interested in? (e.g., RPG, Horror, Puzzle)",
                    "game": None
                }

            # Step 2: Capture genre
            if survey_session["genre"] == "pending":
                survey_session["genre"] = user_msg
                return {
                    "response": "Great! What's your budget? (e.g., 10, 20, 30)",
                    "game": None
                }

            # Step 3: Capture budget
            if survey_session["budget"] is None:
                try:
                    survey_session["budget"] = float(user_msg)
                except ValueError:
                    return {
                        "response": "Please enter a valid number for your budget.",
                        "game": None
                    }
                return {
                    "response": "Do you prefer highly rated games (≥80% positive)? Reply with 'yes' or 'any'.",
                    "game": None
                }

            # Step 4: Capture rating preference and filter games
            if survey_session["rating_preference"] is None:
                if user_msg.lower() in ["yes", "y", "prefer"]:
                    survey_session["rating_preference"] = (80, "Very Positive")
                else:
                    survey_session["rating_preference"] = ("any", "any")

                genre = survey_session["genre"]
                budget = survey_session["budget"]
                min_ratio, min_rating = survey_session["rating_preference"]

                filtered = filter_by_genre(ALL_GAMES, genre)
                filtered = filter_by_budget(filtered, budget)
                filtered = filter_by_rating(filtered, min_rating=min_rating, min_ratio=min_ratio)

                if not filtered:
                    SWITCH_INTENTS = True
                    survey_session = reset_survey()
                    return {
                        "response": "Sorry, I couldn't find any games matching your preferences.",
                        "game": None
                    }

                survey_session["filtered_games"] = filtered
                survey_session["current_index"] = 0
                survey_session["confirmation"] = "show"

            if survey_session["confirmation"] == "show":
                current_game = survey_session["filtered_games"][survey_session["current_index"]]
                survey_session["confirmation"] = "pending"
                return {
                    "response": f"How about this one: {current_game['title']}?\n\nIs this the game you want? (yes/no)",
                    "game": current_game
                }

            if survey_session["confirmation"] == "pending":
                if user_msg.lower() in ["yes", "y"]:
                    best_game = survey_session["filtered_games"][survey_session["current_index"]]
                    game_data = get_game_recommendation(best_game["app_id"])

                    response_msg = f"Here's the summary of the game:\n\n{game_data['processed_description']}"

                    SWITCH_INTENTS = True
                    survey_session = reset_survey()

                    return {
                        "response": response_msg,
                        "game": game_data
                    }

                elif user_msg.lower() in ["no", "n", "next"]:
                    survey_session["current_index"] += 1
                    if survey_session["current_index"] >= len(survey_session["filtered_games"]):
                        SWITCH_INTENTS = True
                        survey_session = reset_survey()
                        return {
                            "response": "That's all I found, no more matching games. Try again with a new genre or budget!",
                            "game": None
                        }

                    survey_session["confirmation"] = "show"
                    return get_reply("show_next", game_data)

        except Exception as e:
            print(f"Error in rule-based recommendation: {e}")
            SWITCH_INTENTS = True
            survey_session = reset_survey()
            return {
                "response": "Something went wrong. Please try again later.",
                "game": None
            }

    chat_reply = generate_chitchat_reply(user_msg)
    return {"response": chat_reply, "game": None}
