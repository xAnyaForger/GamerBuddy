import customtkinter as ctk
import threading
from game import get_reply, TextProcessor
import webbrowser
import requests
from io import BytesIO
from PIL import Image
import time 

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GameChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Game Recommendation Chat")
        self.geometry("1560x900")
        self.minsize(1200, 700)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=700)
        
        self.left_sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_sidebar.grid(row=0, rowspan=2, column=0, sticky="nsew", padx=(10,5), pady=10)
        self.left_sidebar.grid_propagate(False)
        
        self.new_chat_btn = ctk.CTkButton(
            self.left_sidebar,
            text="+ New Chat",
            command=self.new_chat,
            fg_color="transparent",
            border_width=1,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.new_chat_btn.pack(pady=(15,10), padx=10, fill="x")
        
        self.chat_list_frame = ctk.CTkScrollableFrame(self.left_sidebar, fg_color="transparent")
        self.chat_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.chat_area = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_area.grid(row=0, column=1, sticky="nsew", padx=5, pady=10, rowspan=2)
        
        self.chat_display = ctk.CTkScrollableFrame(self.chat_area, fg_color="transparent", width=600)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.input_frame = ctk.CTkFrame(self.chat_area)
        self.input_frame.pack(fill="x", padx=10, pady=(0,10))
        
        self.chat_input = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="What kind of games are you looking for?",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0,10))
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=80,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.send_message
        )
        self.send_button.pack(side="right")
        
        self.right_sidebar = ctk.CTkFrame(self, width=400, corner_radius=0)
        self.right_sidebar.grid(row=0, column=2, sticky="nsew", padx=(5,10), pady=10)
        self.right_sidebar.grid_propagate(False)

        self.right_sidebar_bottom = ctk.CTkFrame(self, width=400, height=100, corner_radius=0)
        self.right_sidebar_bottom.grid(row=1, column=2, sticky="nsew", padx=(5,10), pady=10)
        self.right_sidebar_bottom.grid_propagate(False)

        self.game_details_frame = ctk.CTkScrollableFrame(
            self.right_sidebar,
            fg_color="transparent",
            scrollbar_button_color="#2b2b2b",
            scrollbar_button_hover_color="#1a5276"
        )
        self.game_details_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.game_title = ctk.CTkLabel(
            self.game_details_frame,
            text="No game selected",
            font=ctk.CTkFont(size=20, weight="bold"),
            wraplength=380
        )
        self.game_title.pack(pady=(0,15), anchor="w")

        self.game_image_label = ctk.CTkLabel(
            self.game_details_frame,
            text="",
            width=380,
            height=180,
            fg_color="#2b2b2b",
            corner_radius=10,
        )
        self.game_image_label.pack(pady=(0,20))
        self.current_image = None

        self.price_frame = ctk.CTkFrame(self.game_details_frame, fg_color="transparent")
        self.price_frame.pack(fill="x", pady=(0,15))

        self.price_label = ctk.CTkLabel(
            self.price_frame,
            text="Price:",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.price_label.pack(side="left")

        self.discounted_price = ctk.CTkLabel(
            self.price_frame,
            text="-",
            font=ctk.CTkFont(size=15)
        )
        self.discounted_price.pack(side="left", padx=5)

        self.original_price = ctk.CTkLabel(
            self.price_frame,
            text="",
            font=ctk.CTkFont(size=14)
        )
        self.original_price.pack(side="left")

        self.desc_frame = ctk.CTkFrame(self.game_details_frame, fg_color="transparent")
        self.desc_frame.pack(fill="x", pady=(0,15))

        ctk.CTkLabel(
            self.desc_frame,
            text="Description:",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w")

        self.desc_text = ctk.CTkLabel(
            self.desc_frame,
            text="Select a game to see details",
            wraplength=380,
            justify="left",
            font=ctk.CTkFont(size=14)
        )
        self.desc_text.pack(anchor="w", pady=(5,0))

        self.rating_frame = ctk.CTkFrame(self.game_details_frame, fg_color="transparent")
        self.rating_frame.pack(fill="x", pady=(0,15))

        ctk.CTkLabel(
            self.rating_frame,
            text="Rating:",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")

        self.rating_value = ctk.CTkLabel(
            self.rating_frame,
            text="-",
            font=ctk.CTkFont(size=15)
        )
        self.rating_value.pack(side="left", padx=5)

        self.release_frame = ctk.CTkFrame(self.game_details_frame, fg_color="transparent")
        self.release_frame.pack(fill="x", pady=(0,15))

        ctk.CTkLabel(
            self.release_frame,
            text="Released:",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")

        self.release_date = ctk.CTkLabel(
            self.release_frame,
            text="-",
            font=ctk.CTkFont(size=15)
        )
        self.release_date.pack(side="left", padx=5)

        self.req_frame = ctk.CTkFrame(self.game_details_frame, fg_color="transparent")
        self.req_frame.pack(fill="x", pady=(0,15))

        ctk.CTkLabel(
            self.req_frame,
            text="Requirements:",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(anchor="w")

        self.requirements = ctk.CTkLabel(
            self.req_frame,
            text="-",
            wraplength=380,
            justify="left",
            font=ctk.CTkFont(size=14)
        )
        self.requirements.pack(anchor="w", pady=(5,0))

        self.store_button = ctk.CTkButton(
            self.game_details_frame,
            text="View on Steam",
            fg_color="#1a9fff",
            hover_color="#0077cc",
            height=35,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.open_steam_store
        )
        self.store_button.pack(fill="x", pady=(10,0))
        self.store_button.pack_forget()
        
        self.status_frame = ctk.CTkFrame(
            self.right_sidebar_bottom,
            fg_color="#2b2b2b",
            corner_radius=10
        )
        self.status_frame.pack(fill="x", padx=10, pady=(0, 10), side="bottom")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Active",
            font=ctk.CTkFont(size=14),
            text_color="#4CAF50"
        )
        self.status_label.pack(pady=(10, 5), padx=10, anchor="w")

        self.uptime_label = ctk.CTkLabel(
            self.status_frame,
            text="Uptime: 00:00:00",
            font=ctk.CTkFont(size=14)
        )
        self.uptime_label.pack(pady=(0, 10), padx=10, anchor="w")

        self.last_activity_time = time.time()
        self.start_time = time.time()
        self.idle_threshold = 60  
        self.is_idle = False

        self.update_uptime()
        self.check_activity()
        
        self.chats = {}
        self.current_chat_id = None
        self.current_game = None
        self.new_chat()

    def update_uptime(self):
        uptime_seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.uptime_label.configure(text=f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.after(1000, self.update_uptime)

    def check_activity(self):
        inactive_time = time.time() - self.last_activity_time
        if inactive_time > self.idle_threshold and not self.is_idle:
            self.set_status("Idle")
        self.after(1000, self.check_activity)

    def set_status(self, status):
        self.is_idle = (status == "Idle")
        color = "#4CAF50" if status == "Active" else "#F44336" 
        self.status_label.configure(text=f"Status: {status}", text_color=color)

    def record_activity(self):
        self.last_activity_time = time.time()
        if self.is_idle:
            self.set_status("Active")

    def new_chat(self):
        self.record_activity()  
        chat_id = f"chat_{len(self.chats)+1}"
        self.chats[chat_id] = {
            "messages": [],
            "widgets": []
        }
        self.current_chat_id = chat_id
        
        btn = ctk.CTkButton(
            self.chat_list_frame,
            text=f"Chat {len(self.chats)}",
            command=lambda id=chat_id: self.switch_chat(id),
            fg_color="transparent",
            anchor="w",
            font=ctk.CTkFont(size=14)
        )
        btn.pack(fill="x", pady=2, padx=5)
        
        for widget in self.chat_display.winfo_children():
            widget.destroy()
        
        self.clear_game_details()
        
        self.add_message(
            "bot",
            "🎮 Welcome to Game Recommendation Chat!\n\n"
            "I can recommend games based on your preferences. "
            "Try asking for:\n"
            "- Co-op games\n"
            "- Open world RPGs\n"
            "- Horror games with multiplayer\n"
            "- Casual games under $20\n\n"
            "What kind of games are you looking for?"
        )
        
        return chat_id
    
    def switch_chat(self, chat_id):
        self.record_activity()
        for widget in self.chat_display.winfo_children():
            widget.destroy()
        
        self.clear_game_details()
        self.current_chat_id = chat_id

        for msg in self.chats[chat_id]["messages"]:
            self.add_message(msg["sender"], msg["text"], from_history=True)
            
            if msg.get("game_data") and msg == self.chats[chat_id]["messages"][-1]:
                self.after(100, lambda: self.show_game_details(msg["game_data"]))
    
    def send_message(self):
        if not self.current_chat_id:
            return
            
        user_msg = self.chat_input.get().strip()
        if not user_msg:
            return
        
        self.record_activity()
        self.add_message("user", user_msg)
        self.chat_input.delete(0, "end")
        
        typing_id = f"typing_{len(self.chats[self.current_chat_id]['messages'])}"
        self.add_message("bot", "Typing...", is_typing=True, typing_id=typing_id)
        
        threading.Thread(
            target=self.get_bot_reply,
            args=(user_msg, typing_id),
            daemon=True
        ).start()
    
    def get_bot_reply(self, user_msg, typing_id):
        try:
            reply_data = get_reply(user_msg, self.current_game)
            
            if isinstance(reply_data, str):
                reply_data = {
                    "response": reply_data,
                    "game": None
                }
                
            self.after(0, lambda: self.update_bot_reply(reply_data, typing_id))
        except Exception as e:
            print(f"Error getting reply: {e}")
            self.after(0, lambda: self.update_bot_reply({
                "response": "Sorry, I couldn't process your request.",
                "game": None
            }, typing_id))
    
    def update_bot_reply(self, reply_data, typing_id):
        for widget in self.chat_display.winfo_children():
            if hasattr(widget, 'typing_id') and widget.typing_id == typing_id:
                widget.destroy()
                break
        
        container = ctk.CTkFrame(self.chat_display, fg_color="transparent")
        container.pack(fill="x", pady=5, padx=5)
        
        label = ctk.CTkLabel(
            container,
            text="",
            wraplength=650,
            justify="left",
            padx=15,
            pady=10,
            font=ctk.CTkFont(size=15),
            anchor="w",
            fg_color="transparent"
        )
        label.pack(side="left", anchor="w")
        
        message_data = {
            "sender": "bot",
            "text": reply_data["response"],
            "game_data": reply_data.get("game")
        }
        self.chats[self.current_chat_id]["messages"].append(message_data)
        
        self.type_message(
            label=label,
            full_text=reply_data["response"],
            speed=50,
            callback=lambda: self.handle_game_data_after_typing(reply_data.get("game"))
        )
        
        self.chat_display._parent_canvas.yview_moveto(1.0)

    def type_message(self, label, full_text, speed=50, callback=None):
        delay = int(1000 / speed) 
        
        def type_next_char(index=0):
            if index < len(full_text):
                current_text = full_text[:index+1]
                label.configure(text=current_text)
                self.after(delay, type_next_char, index+1)
            elif callback:
                callback()
        
        type_next_char()

    def handle_game_data_after_typing(self, game_data):
        if game_data:
            self.show_game_details(game_data)
    
    def add_message(self, sender, text, is_typing=False, typing_id=None, from_history=False, game_data=None):
        container = ctk.CTkFrame(self.chat_display, fg_color="transparent")
        container.pack(fill="x", pady=5, padx=5)
        
        if sender == "bot":
            label = ctk.CTkLabel(
                container,
                text=text,
                wraplength=650,
                justify="left",
                padx=15,
                pady=10,
                font=ctk.CTkFont(size=15),
                anchor="w",
                fg_color="transparent"
            )
            label.pack(side="left", anchor="w")
        else:
            frame = ctk.CTkFrame(
                container,
                corner_radius=10,
                fg_color="#1a5276"
            )
            frame.pack(side="right", anchor="e")
            
            label = ctk.CTkLabel(
                frame,
                text=text,
                wraplength=650,
                justify="left",
                padx=15,
                pady=10,
                font=ctk.CTkFont(size=15)
            )
            label.pack()
        
        if is_typing:
            container.typing_id = typing_id
        
        if not from_history and self.current_chat_id:
            self.chats[self.current_chat_id]["messages"].append({
                "sender": sender,
                "text": text,
                "game_data": game_data
            })
        
        self.chat_display._parent_canvas.yview_moveto(1.0)
    
    def load_steam_image(self, app_id):
        try:
            url = f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"
            response = requests.get(url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            image = image.resize((380, 180), Image.LANCZOS)
            return ctk.CTkImage(light_image=image, dark_image=image, size=(380, 180))
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def clear_game_details(self):
        self.current_game = None
        
        self.game_title.configure(text="No game selected")
        
        self.game_image_label.configure(image=None)
        self.game_image_label._text = ""
        if hasattr(self.game_image_label, '_label'):
            self.game_image_label._label.configure(image='', text="")
        
        self.original_price.configure(text="")
        self.discounted_price.configure(text="-")
        self.desc_text.configure(text="Select a game to see details")
        self.rating_value.configure(text="-")
        self.release_date.configure(text="-")
        self.requirements.configure(text="-")
        self.store_button.pack_forget()
        
        self.update_idletasks()

    def show_game_details(self, game_data):
        self.clear_game_details()
        self.current_game = game_data
        
        self.game_title.configure(
            text=game_data.get("title", "Unknown Game"),
            wraplength=380
        )
        
        if "app_id" in game_data:
            try:
                self.game_image_label.configure(image=None)
                image = self.load_steam_image(game_data["app_id"])
                if image:
                    self.current_image = image  
                    self.game_image_label.configure(image=image)
                    self.game_image_label._text = ""
                    self.game_image_label._label.configure(text="")
                else:
                    self.game_image_label.configure(image=None)
                    self.game_image_label._text = "🎮"
                    self.game_image_label._label.configure(text="🎮")
            except Exception as e:
                print(f"Error setting image: {e}")
                self.game_image_label.configure(image=None)
                self.game_image_label._text = "🎮"
                self.game_image_label._label.configure(text="🎮")
        else:
            self.game_image_label.configure(image=None)
            self.game_image_label._text = "🎮"
            self.game_image_label._label.configure(text="🎮")
        
        price_original = game_data.get("price_original", 0)
        price_final = game_data.get("price_final", price_original)
        
        if price_original > price_final: 
            self.original_price.configure(
                text=f"~~${price_original:.2f}~~",
                text_color="#aaaaaa"
            )
            self.discounted_price.configure(
                text=f"${price_final:.2f}",
                text_color="#ffffff"
            )
        else:
            self.original_price.configure(text="")
            self.discounted_price.configure(
                text=f"${price_final:.2f}",
                text_color="#ffffff"
            )
        
        description = game_data.get("description", 
                                game_data.get("description", "No description available"))
        self.desc_text.configure(
            text=description,
            wraplength=380
        )
        
        rating = game_data.get("rating", "")
        positive_ratio = game_data.get("positive_ratio", "")
        user_reviews = game_data.get("user_reviews", "")
        
        rating_text = rating
        if positive_ratio:
            rating_text += f" ({positive_ratio}%)"
        if user_reviews:
            rating_text += f" from {user_reviews} reviews"
        
        self.rating_value.configure(
            text=rating_text if rating_text else "Not rated"
        )
        
        platforms = []
        if game_data.get("win"):
            platforms.append("Windows")
        if game_data.get("mac"):
            platforms.append("Mac")
        if game_data.get("linux"):
            platforms.append("Linux")
        if game_data.get("steam_deck"):
            platforms.append("Steam Deck")
        
        self.release_date.configure(
            text=", ".join(platforms) if platforms else "Platform not specified"
        )
        
        tags = game_data.get("tags", [])
        self.requirements.configure(
            text="Tags: " + ", ".join(tags) if tags else "No tags available",
            wraplength=380
        )
        
        if "app_id" in game_data:
            self.store_button.configure(text="View on Steam")
            self.store_button.pack(pady=(10,0))
        else:
            self.store_button.pack_forget()
        
        self.update_idletasks()
        self.game_details_frame._parent_canvas.yview_moveto(0)

    def open_steam_store(self):
        if self.current_game and "app_id" in self.current_game:
            webbrowser.open(f"https://store.steampowered.com/app/{self.current_game['app_id']}")
        
        def open_steam_store(self):
            if self.current_game and "store_url" in self.current_game:
                webbrowser.open(self.current_game["store_url"])

if __name__ == "__main__":
    processor = TextProcessor()
    app = GameChatApp()
    app.mainloop()