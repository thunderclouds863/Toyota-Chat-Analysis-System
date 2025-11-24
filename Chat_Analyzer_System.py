# Setup & Configuration
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import json
import os
import joblib
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # File paths
    RAW_DATA_PATH = "data/raw_conversation.xlsx"
    PROCESSED_DATA_PATH = "data/processed/conversations.pkl"
    MODEL_SAVE_PATH = "models/"
    
    # ML Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Time thresholds (dalam menit)
    NORMAL_THRESHOLD = 5
    SERIOUS_FIRST_REPLY_THRESHOLD = 5
    SERIOUS_FINAL_REPLY_THRESHOLD = 480  # 8 jam
    COMPLAINT_FINAL_REPLY_THRESHOLD = 7200  # 5 hari
    
    # Abandoned detection
    ABANDONED_TIMEOUT_MINUTES = 3  # 3 menit tanpa response dari customer
    CUSTOMER_LEAVE_TIMEOUT = 3  # 3 menit untuk detect customer leave
    
    # Follow-up settings
    FOLLOWUP_SOURCES = {
        'phone_sources': [
            '[ Dolphin Live Chat ] New Livechat mToyota',
            '[ Dolphin Live Chat ] TAM LiveChat'
        ],
        'name_sources': [
            '[ Instagram Business ] toyotaid',
            '[ Instagram DM ] toyotaid' 
        ],
        'ignore_sources': [
            '[ Facebook Messenger ] ToyotaID'
        ]
    }

config = Config()

# Create directories
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("output/reports").mkdir(parents=True, exist_ok=True)
Path("output/visualizations").mkdir(parents=True, exist_ok=True)

print("✅ Setup completed!")

# Data Preprocessor
class DataPreprocessor:
    def __init__(self):
        self.role_mapping = {
            'bot': 'Bot',
            'customer': 'Customer', 
            'operator': 'Operator',
            'ticket automation': 'Ticket Automation'
        }
    
    def load_raw_data(self, file_path):
        """Load data dari Excel dengan format yang ditentukan"""
        print(f"📖 Loading data from {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Loaded {len(df)} rows")
            
            # Validasi columns
            required_columns = ['No', 'Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️ Missing columns: {missing_columns}")
                return None
            
            # Clean data
            df = self.clean_data(df)
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean dan preprocess data"""
        # Copy dataframe
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.dropna(subset=['Message', 'Ticket Number'])
        
        # Clean text
        df_clean['Message'] = df_clean['Message'].astype(str).str.strip()
        
        # Parse timestamp
        df_clean['parsed_timestamp'] = pd.to_datetime(
            df_clean['Message Date'], errors='coerce'
        )
        
        # Remove invalid timestamps
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['parsed_timestamp'].notna()]
        final_count = len(df_clean)
        print(f"📅 Valid timestamps: {final_count}/{initial_count}")
        
        # Standardize roles
        df_clean['Role'] = df_clean['Role'].str.lower().map(
            lambda x: self.role_mapping.get(x, x.title())
        )
        
        # Filter meaningful messages
        df_clean = df_clean[df_clean['Message'].str.len() > 1]
        
        print(f"🧹 Cleaned data: {len(df_clean)} rows")
        return df_clean
    
    def extract_customer_info(self, df):
        """Extract customer phone dan name dari setiap ticket"""
        customer_info = {}
        
        for ticket_id in df['Ticket Number'].unique():
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            # Cari di row pertama ticket untuk phone/name
            first_row = ticket_df.iloc[0]
            
            # Check semua kolom untuk phone/name patterns
            phone = None
            name = None
            
            for col in ticket_df.columns:
                if pd.api.types.is_string_dtype(ticket_df[col]):
                    # Cari phone patterns
                    phone_patterns = [r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']
                    for pattern in phone_patterns:
                        matches = ticket_df[col].astype(str).str.extract(f'({pattern})', expand=False)
                        if not matches.isna().all():
                            phone = matches.dropna().iloc[0] if not matches.dropna().empty else None
                            break
                    
                    # Cari name patterns (asumsi di kolom Sender atau kolom teks lainnya)
                    if 'sender' in col.lower() or 'name' in col.lower():
                        name_candidates = ticket_df[col].astype(str).str.extract(r'([A-Za-z\s]{3,})', expand=False)
                        if not name_candidates.isna().all():
                            name = name_candidates.dropna().iloc[0] if not name_candidates.dropna().empty else None
            
            customer_info[ticket_id] = {
                'phone': phone,
                'name': name,
                'source': first_row.get('Source', 'Unknown') if 'Source' in ticket_df.columns else 'Unknown'
            }
        
        return customer_info

# ===== CONVERSATION PARSER - FIXED VERSION =====
class ConversationParser:
    def __init__(self):
        self.question_indicators = [
            '?', 'apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa',
            'bisa', 'boleh', 'minta', 'tolong', 'tanya', 'info', 'caranya',
            'mau tanya', 'boleh tanya', 'minta info', 'berapa harga',
            'bagaimana cara', 'bisa tolong', 'mohon bantuan', 'gimana',
            'promo', 'error', 'rusak', 'masalah', 'mogok', 'gagal', 'tidak bisa',
            'harga', 'biaya', 'tarif', 'fungsi', 'cara', 'solusi', 'bantuan',
            'mobil', 'servis', 'booking', 'test drive', 'dealer', 'bengkel',
            'sparepart', 'oli', 'ban', 'aki', 'mesin', 'rem', 'transmisi'
        ]
        
        self.operator_greeting_patterns = [
            r"selamat\s+(pagi|siang|sore|malam)",
            r"selamat\s+\w+\s+selamat\s+datang",
            r"selamat\s+datang",
            r"dengan\s+\w+\s+apakah\s+ada",
            r"ada\s+yang\s+bisa\s+dibantu",
            r"boleh\s+dibantu",
            r"bisa\s+dibantu", 
            r"halo.*selamat",
            r"hai.*selamat",
            r"perkenalkan.*saya",
            r"layanan\s+live\s+chat",
            r"live\s+chat\s+toyota",
            r"toyota\s+astra\s+motor"
        ]
        
        self.bot_patterns = [
            'klik.*setuju', 'data privasi', 'virtual assistant', 'silakan memilih',
            'pilih menu', 'silahkan ketik nama', 'halo kak', 'bucket', 'media-images',
            'pusat bantuan', 'main menu', 'feedback'
        ]
        
        self.generic_reply_patterns = [
            'terima kasih telah menghubungi',
            'silakan pilih menu',
            'virtual assistant',
            'akan segera menghubungi',
            'dalam antrian',
            'tunggu sebentar',
            'terima kasih, saat ini anda masuk',
            'customer service akan',
            'menghubungi anda'
        ]
        
        self.conversation_ender_patterns = [
            'apakah sudah cukup',
            'apakah informasinya sudah cukup jelas',
            'terima kasih',
            'sampai jumpa',
            'goodbye'
        ]
        
        self.customer_leave_timeout = 3  # 3 menit
    
    def detect_conversation_start(self, ticket_df):
        """Deteksi kapan conversation benar-benar dimulai dengan operator"""
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        print(f"   🔍 Analyzing {len(ticket_df)} messages for conversation start...")
        
        # METHOD 1: Cari operator greeting message
        for idx, row in ticket_df.iterrows():
            message = str(row['Message']).lower()
            role = str(row['Role']).lower()
            
            if self._is_bot_message(message, role):
                continue
                
            if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                for pattern in self.operator_greeting_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        print(f"   ✅ Conversation start: operator greeting at position {idx}")
                        return row['parsed_timestamp']
        
        # METHOD 2: Cari first operator response to meaningful customer question
        meaningful_questions = []
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message']).lower()
            
            if self._is_bot_message(message, role):
                continue
                
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                if self._is_meaningful_message(message):
                    meaningful_questions.append({
                        'index': idx, 'time': row['parsed_timestamp'], 'message': message
                    })
            
            elif meaningful_questions and any(keyword in role for keyword in ['operator', 'agent']):
                last_question = meaningful_questions[-1]
                time_gap = (row['parsed_timestamp'] - last_question['time']).total_seconds()
                
                if time_gap < 180:  # 3 menit
                    print(f"   ✅ Conversation start: first operator response at position {idx}")
                    return last_question['time']
        
        # Fallback methods
        if meaningful_questions:
            print(f"   ⚠️  Conversation start: first meaningful question")
            return meaningful_questions[0]['time']
            
        for idx, row in ticket_df.iterrows():
            message = str(row['Message']).lower()
            role = str(row['Role']).lower()
            
            if not self._is_bot_message(message, role):
                print(f"   ⚠️  Conversation start: first non-bot message")
                return row['parsed_timestamp']
        
        if len(ticket_df) > 0:
            print(f"   ⚠️  Conversation start: first message")
            return ticket_df.iloc[0]['parsed_timestamp']
        
        print("   ❌ No conversation start detected")
        return None
    
    def detect_customer_leave(self, ticket_df):
        """Deteksi jika customer meninggalkan conversation - IMPROVED"""
        ticket_df = ticket_df.sort_values('parsed_timestamp')
        
        last_customer_time = None
        last_operator_time = None
        conversation_end_detected = False
        
        for _, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message']).lower()
            
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                last_customer_time = row['parsed_timestamp']
            
            elif any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                last_operator_time = row['parsed_timestamp']
                
                # Check jika operator mengakhiri conversation
                if any(ender in message for ender in self.conversation_ender_patterns):
                    conversation_end_detected = True
        
        # Jika operator mengakhiri conversation, bukan customer leave
        if conversation_end_detected:
            return False
        
        # 🔥 LOGIC BARU: Jika customer tidak response dalam 3 menit setelah operator message terakhir
        if last_customer_time and last_operator_time:
            time_gap = (last_operator_time - last_customer_time).total_seconds() / 60
            
            # Customer leave jika: 
            # 1. Operator message terakhir SETELAH customer message terakhir
            # 2. Gap waktu > 3 menit
            # 3. Tidak ada conversation ender
            if last_operator_time > last_customer_time and time_gap >= self.customer_leave_timeout:
                print(f"   🚶 Customer leave detected: {time_gap:.1f} min gap")
                return True
        
        return False
    
    def parse_conversation(self, ticket_df):
        """Parse conversation menjadi Q-A pairs - RELAXED VERSION"""
        conversation_start = self.detect_conversation_start(ticket_df)
        customer_leave = self.detect_customer_leave(ticket_df)
        
        if not conversation_start:
            print("   ⚠️  Using all non-bot messages")
            conv_df = self._filter_bot_messages(ticket_df)
        else:
            conv_df = self._filter_bot_messages(ticket_df)
            conv_df = conv_df[conv_df['parsed_timestamp'] >= conversation_start]
        
        print(f"   📝 Analyzing {len(conv_df)} meaningful messages")
        print(f"   🚶 Customer leave detected: {customer_leave}")
        
        if len(conv_df) == 0:
            print("   ❌ No meaningful messages after filtering")
            return []
        
        # 🔥 FIX: URUTKAN DATA BERDASARKAN TIMESTAMP DULU!
        conv_df = conv_df.sort_values('parsed_timestamp').reset_index(drop=True)
        print(f"   🔄 Sorted messages by timestamp")
        
        qa_pairs = []
        current_question = None
        question_time = None
        question_context = []
        last_customer_time = None
        
        # Track roles untuk debugging
        customer_count = 0
        operator_count = 0
        
        for idx, row in conv_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            timestamp = row['parsed_timestamp']
            
            # CUSTOMER MESSAGE - RELAXED DETECTION
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                customer_count += 1
                last_customer_time = timestamp
                
                # RELAXED: Hampir semua customer message dianggap meaningful question
                if self._is_relaxed_meaningful_message(message):
                    # Jika ada previous question yang belum dijawab, simpan dulu
                    if current_question and question_context:
                        self._save_qa_pair(qa_pairs, question_context, question_time, None, None, position=idx)
                    
                    # Start new question
                    current_question = message
                    question_time = timestamp
                    question_context = [message]
                    print(f"   💬 Customer question {customer_count}: {message[:50]}...")
                
                elif current_question and question_context:
                    # Check jika ini bagian dari bubble chat yang sama
                    time_gap = (timestamp - question_time).total_seconds()
                    if time_gap < 600:  # 10 menit (diperlonggar)
                        question_context.append(message)
                        question_time = timestamp  # Update ke timestamp terakhir
                        print(f"   💬 Bubble chat added: {message[:30]}...")
            
            # OPERATOR MESSAGE - potential answer (RELAXED)
            elif current_question and question_context and any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                operator_count += 1
                answer = message
                
                # RELAXED: Hampir semua operator message dianggap potential answer
                # Skip hanya yang benar-benar generic
                if self._is_generic_reply(answer):
                    print(f"   ⏭️  Skipping very generic reply: {answer[:30]}...")
                    continue
                
                # Pastikan ini benar-benar jawaban (bukan greeting awal)
                time_gap = (timestamp - question_time).total_seconds()
                
                # RELAXED: Boleh 0 atau positif (setelah question)
                if time_gap >= 0:  
                    lead_time = time_gap
                    self._save_qa_pair(qa_pairs, question_context, question_time, answer, timestamp, role, lead_time, position=idx)
                    print(f"   ✅ Operator answer {operator_count}: {answer[:50]}... (LT: {lead_time/60:.1f}min)")
                    
                    # Reset untuk next question
                    current_question = None
                    question_time = None
                    question_context = []
        
        # Handle last question jika ada
        if current_question and question_context:
            self._save_qa_pair(qa_pairs, question_context, question_time, None, None, position=len(conv_df))
            print(f"   ❓ Unanswered question: {current_question[:50]}...")
        
        # POST-PROCESSING: Cari jawaban untuk unanswered questions (AGGRESSIVE)
        qa_pairs = self._find_missing_answers_aggressive(conv_df, qa_pairs)
        
        # 🔥 FIX: URUTKAN Q-A PAIRS BERDASARKAN QUESTION TIME!
        qa_pairs = sorted(qa_pairs, key=lambda x: x['question_time'] if x['question_time'] else pd.Timestamp.min)
        
        # 🔥 FIX: TAMBAHKAN POSITION INDEX YANG BENAR BERDASARKAN URUTAN WAKTU
        for i, pair in enumerate(qa_pairs):
            pair['position'] = i  # Position berdasarkan urutan waktu
            pair['customer_leave'] = customer_leave  # Tambah flag customer leave
        
        print(f"   📊 Conversation stats: {customer_count} customer msgs, {operator_count} operator msgs")
        print(f"   ✅ Found {len(qa_pairs)} Q-A pairs ({sum(1 for p in qa_pairs if p['is_answered'])} answered)")
        
        return qa_pairs

    def _is_relaxed_meaningful_message(self, message):
        """RELAXED version - hampir semua customer message dianggap meaningful"""
        if not message or len(message.strip()) < 2:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip very short messages yang cuma greetings/consent
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam']
        consent_words = ['setuju', 'consent', 'agree', 'ok', 'oke', 'iya', 'yes']
        menu_words = ['menu', 'pusatbantuan', 'mainmenu', 'kembali', 'others']
        
        words = message_lower.split()
        if len(words) <= 2:
            if any(word in words for word in greetings + consent_words + menu_words):
                return False
        
        # Skip pure consent messages
        if any(word in message_lower for word in consent_words) and len(message_lower) < 10:
            return False
            
        # Skip menu navigation
        if any(word in message_lower for word in menu_words) and len(message_lower) < 15:
            return False
        
        # RELAXED: Hampir semua message dengan > 3 kata dianggap meaningful
        word_count = len(message_lower.split())
        if word_count >= 3:
            return True
        
        # Question indicators (relaxed)
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_question_mark = '?' in message_lower
        
        return has_question_indicator or has_question_mark
    
    def _is_generic_reply(self, message):
        """Hanya skip yang benar-benar generic"""
        message_lower = str(message).lower()
        
        very_generic_patterns = [
            r'virtual\s+assistant',
            r'akan\s+segera\s+menghubungi', 
            r'dalam\s+antrian',
            r'terima\s+kasih,\s+saat\s+ini\s+anda\s+masuk',
            r'customer\s+service\s+akan',
            r'menghubungi\s+anda',
            r'silakan\s+memilih\s+dari\s+menu',
            r'klik\s+setuju',
            r'data\s+privasi',
            r'pilih\s+menu',
            r'silahkan\s+ketik\s+nama'
        ]

        return any(re.search(pattern, message_lower) for pattern in very_generic_patterns)
    
    def _find_missing_answers_aggressive(self, conv_df, qa_pairs):
        """AGGRESSIVE: Cari jawaban untuk questions yang belum terjawab"""
        answered_pairs = [p for p in qa_pairs if p['is_answered']]
        unanswered_pairs = [p for p in qa_pairs if not p['is_answered']]
        
        if not unanswered_pairs:
            return qa_pairs
        
        print(f"   🔍 Aggressive search for {len(unanswered_pairs)} unanswered questions")
        
        for i, pair in enumerate(unanswered_pairs):
            question_time = pair['question_time']
            
            # Cari SEMUA operator messages setelah question time
            subsequent_messages = conv_df[
                (conv_df['parsed_timestamp'] > question_time) &
                (conv_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
            ].sort_values('parsed_timestamp')
            
            if not subsequent_messages.empty:
                # Ambil operator message pertama setelah question yang tidak terlalu generic
                for _, operator_msg in subsequent_messages.iterrows():
                    if not self._is_generic_reply(operator_msg['Message']):
                        lead_time = (operator_msg['parsed_timestamp'] - question_time).total_seconds()
                        
                        # Update pair dengan answer yang ditemukan
                        pair.update({
                            'answer': operator_msg['Message'],
                            'answer_time': operator_msg['parsed_timestamp'],
                            'answer_role': operator_msg['Role'],
                            'lead_time_seconds': lead_time,
                            'lead_time_minutes': round(lead_time / 60, 2),
                            'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                            'is_answered': True
                        })
                        print(f"   🔍 Found aggressive answer for question {i+1}: {operator_msg['Message'][:50]}...")
                        break
        
        return answered_pairs + unanswered_pairs

    def _save_qa_pair(self, qa_pairs, question_context, question_time, answer, answer_time, answer_role=None, lead_time=None, position=None):
        """Save Q-A pair ke list - FIXED dengan position"""
        full_question = " | ".join(question_context)
        
        pair_data = {
            'question': full_question,
            'question_time': question_time,
            'bubble_count': len(question_context),
            'is_answered': answer is not None,
            'position': position if position is not None else len(qa_pairs)  # Default position
        }
        
        if answer:
            pair_data.update({
                'answer': answer,
                'answer_time': answer_time,
                'answer_role': answer_role,
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2) if lead_time else None,
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time) if lead_time else None
            })
        else:
            pair_data.update({
                'answer': 'NO_ANSWER',
                'answer_time': None,
                'answer_role': None,
                'lead_time_seconds': None,
                'lead_time_minutes': None,
                'lead_time_hhmmss': None
            })
        
        qa_pairs.append(pair_data)
    
    def _filter_bot_messages(self, df):
        """Filter out bot messages dari dataframe"""
        return df[~df.apply(
            lambda row: self._is_bot_message(str(row['Message']).lower(), str(row['Role']).lower()), 
            axis=1
        )].copy()
    
    def _is_bot_message(self, message, role):
        """Check jika message dari bot"""
        if any(keyword in role for keyword in ['bot', 'system', 'virtual', 'automation']):
            return True
        return any(pattern in message for pattern in self.bot_patterns)
    
    def _is_meaningful_message(self, message):
        """Check jika message meaningful untuk analysis"""
        if not message or len(message.strip()) < 3:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip very short messages yang cuma greetings/consent
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam']
        consent_words = ['setuju', 'consent', 'agree', 'ok', 'oke', 'iya', 'yes']
        menu_words = ['menu', 'pusatbantuan', 'mainmenu', 'kembali', 'others']
        
        words = message_lower.split()
        if len(words) <= 2:
            if any(word in words for word in greetings + consent_words + menu_words):
                return False
        
        # Skip pure consent messages
        if any(word in message_lower for word in consent_words) and len(message_lower) < 10:
            return False
            
        # Skip menu navigation
        if any(word in message_lower for word in menu_words) and len(message_lower) < 15:
            return False
        
        # Question indicators
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_question_mark = '?' in message_lower
        
        # Meaningful content check
        meaningful_words = [w for w in words if len(w) > 2 and w not in greetings + consent_words + menu_words]
        has_meaningful_content = len(meaningful_words) >= 2
        
        return (has_question_indicator and has_meaningful_content) or has_question_mark or len(meaningful_words) >= 3
    
    def _is_generic_reply(self, message):
        """Skip generic/bot replies"""
        message_lower = str(message).lower()
        return any(pattern in message_lower for pattern in self.generic_reply_patterns)
    
    def _seconds_to_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "00:00:00"

# Main Issue Detector dengan Enhanced Question Detection
class MainIssueDetector:
    def __init__(self):
        self.issue_keywords = {
            'serious': [
                'error', 'rusak', 'masalah', 'gagal', 'mogok', 'mati', 'tidak bisa', 
                'help', 'urgent', 'kendala', 'trouble', 'macet', 'hang', 'blank',
                'not responding', 'bermasalah', 'gangguan', 'mogok', 'starter',
                'rem blong', 'overheating', 'transmisi', 'kelistrikan', 'aki soak'
            ],
            'complaint': [
                'lagi servis', 'setelah servis', 'ban mobil pecah', 'pecah ban', 'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan', 
                'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa', 'refund',
                'garansi ditolak', 'pelayanan buruk', 'tidak profesional', 'minta uang kembali',
                'kompensasi', 'ganti rugi', 'uang kembali', 'ulah montir', 'montir', 'teknisinya',  # TAMBAHAN
                'rusak karena', 'akibat', 'kesalahan', 'kecerobohan', 'kerusakan akibat'
            ],
            'normal': [
                'tanya', 'info', 'harga', 'berapa', 'cara', 'bagaimana', 'fungsi', 
                'promo', 'spesifikasi', 'fitur', 'mau tanya', 'boleh tanya', 
                'minta info', 'informasi', 'tanyakan', 'booking', 'test drive',
                'alamat', 'lokasi', 'jam operasional', 'sparepart'
            ]
        }
        
        self.compensation_patterns = [
            r'kompensasi.*ulah montor',
            r'kompensasi.*montir',
            r'kompensasi.*teknisinya', 
            r'ganti rugi.*rusak',
            r'uang kembali.*rusak',
            r'rusak.*akibat.*montir',
            r'ulah montor.*kompensasi'
        ]
        
        # initial question indicators
        self.initial_question_indicators = [
            'mau tanya', 'boleh tanya', 'minta info', 'tanya dong', 'berapa harga',
            'bagaimana cara', 'info', 'promo', 'spesifikasi', 'fungsi',
            'apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa',  
            'bisa', 'boleh', 'minta', 'tolong', 'caranya', 'gimana'    
        ]
        
        # follow-up indicators 
        self.follow_up_indicators = [
            'ok', 'oke', 'baik', 'sip', 'terima kasih', 'tks', 'thanks', 'makasih',
            'kalau', 'jadi', 'berarti', 'apakah', 'apakah benar', 'bukan', 'oh',
            'clarification', 'follow up', 'lanjutan' 
        ]
    
    def _is_compensation_context(self, message):
        """Deteksi konteks kompensasi/komplain"""
        message_lower = message.lower()
        
        # Check untuk pattern kompensasi spesifik
        compensation_indicators = [
            'kompensasi', 'ganti rugi', 'uang kembali', 'minta ganti', 
            'rusak karena', 'akibat montir', 'ulah montor', 'kesalahan teknis'
        ]
        
        # Check jika ada kata-kata kerusakan + kompensasi
        damage_words = ['rusak', 'pecah', 'patah', 'hilang', 'sobek', 'cacat']
        compensation_words = ['kompensasi', 'ganti rugi', 'uang kembali', 'minta ganti']
        
        has_damage = any(word in message_lower for word in damage_words)
        has_compensation = any(word in message_lower for word in compensation_words)
        
        return (has_damage and has_compensation) or any(indicator in message_lower for indicator in compensation_indicators)

    def _is_initial_question(self, question):
        """Improved initial question detection"""
        question_lower = question.lower()
        
        # 1. Check explicit initial question patterns
        if any(indicator in question_lower for indicator in self.initial_question_indicators):
            return True
        
        # 2. Check jika question diawali dengan question word 
        question_words = ['apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa', 'bisa', 'boleh']
        first_word = question_lower.split()[0] if question_lower.split() else ""
        if first_word in question_words:
            return True
        
        # 3. Check question mark dan meaningful content 
        has_question_mark = '?' in question_lower
        word_count = len(question_lower.split())
        
        if has_question_mark and word_count >= 4 and not self._is_follow_up(question):
            return True
        
        return False
    
    def _is_follow_up(self, question):
        """Follow-up detection"""
        question_lower = question.lower()
        
        # 1. Check explicit follow-up patterns
        if any(indicator in question_lower for indicator in self.follow_up_indicators):
            return True
        
        # 2. Check jika question pendek dan mengandung acknowledgment
        words = question_lower.split()
        acknowledgment_words = ['ok', 'oke', 'baik', 'sip', 'tks', 'thanks']
        
        if len(words) <= 3 and any(word in acknowledgment_words for word in words):
            return True
        
        # 3. Check clarification patterns
        clarification_indicators = ['berarti', 'jadi', 'bukan', 'oh', 'apakah benar']
        if any(indicator in question_lower for indicator in clarification_indicators):
            return True
        
        return False
    
    def detect_main_issue(self, qa_pairs):
        """Deteksi main issue dari Q-A pairs - IMPROVED"""
        if not qa_pairs:
            return None
        
        scored_issues = []
        
        for i, pair in enumerate(qa_pairs):
            if not pair['is_answered']:
                continue
                
            question = pair['question'].lower()
            score = 0

            complaint_matches = 0
            serious_matches = 0
            normal_matches = 0

            if self._is_compensation_context(pair['question']):
                score += 5  # Bonus besar untuk konteks kompensasi
                complaint_matches += 3  # Force complaint classification
            
            # 1. Keyword-based scoring 
            complaint_matches += sum(1 for kw in self.issue_keywords['complaint'] if kw in question)
            serious_matches += sum(1 for kw in self.issue_keywords['serious'] if kw in question) 
            normal_matches += sum(1 for kw in self.issue_keywords['normal'] if kw in question)
            
            # Weighted scoring - complaint > serious > normal
            score += (complaint_matches * 3) + (serious_matches * 2) + (normal_matches * 1)
            
            # 2. Question type analysis 
            is_initial_question = self._is_initial_question(pair['question'])  
            is_follow_up = self._is_follow_up(pair['question'])  
            
            if is_initial_question and not is_follow_up:
                score += 3  # Strong bonus untuk initial question
            elif is_follow_up and not is_initial_question:
                score -= 2  # Penalty untuk follow-up/clarification questions
            
            # 3. Position scoring 
            if i == 0:  # First question
                score += 2
            elif i == 1:  # Second question  
                score += 1
            
            # 4. Content quality scoring 
            # Question yang lebih panjang dan detailed biasanya lebih important
            word_count = len(question.split())
            if word_count > 8:
                score += 1
            elif word_count < 4:
                score -= 1
            
            # 5. Lead time consideration 
            # Questions dengan lead time sangat panjang mungkin important
            lead_time = pair.get('lead_time_minutes', 0)
            if lead_time > 15:  # Lebih dari 15 menit
                score += 1
            
            scored_issues.append({
                'question': pair['question'],
                'question_time': pair['question_time'],
                'score': max(score, 0),  
                'position': i,
                'lead_time': lead_time,
                'bubble_count': pair.get('bubble_count', 1),
                'word_count': word_count,
                'is_initial_question': is_initial_question,
                'is_follow_up': is_follow_up,
                'complaint_matches': complaint_matches,
                'serious_matches': serious_matches, 
                'normal_matches': normal_matches,
                'pair_data': pair
            })
        
        if not scored_issues:
            return None
        
        # Pilih question dengan score tertinggi
        main_issue = max(scored_issues, key=lambda x: x['score'])
        
        # IMPROVED issue type determination
        if main_issue['complaint_matches'] > 0:
            issue_type = 'complaint'
        elif main_issue['serious_matches'] > 0:
            issue_type = 'serious'
        elif main_issue['normal_matches'] > 0:
            issue_type = 'normal'
        else:
            # Fallback based on score
            if main_issue['score'] >= 5:
                issue_type = 'complaint'
            elif main_issue['score'] >= 3:
                issue_type = 'serious'
            else:
                issue_type = 'normal'
        
        # Build detailed reason
        reason_parts = []
        if main_issue['is_initial_question']:
            reason_parts.append("initial question")
        if main_issue['complaint_matches'] > 0:
            reason_parts.append(f"{main_issue['complaint_matches']} complaint keywords")
        if main_issue['serious_matches'] > 0:
            reason_parts.append(f"{main_issue['serious_matches']} serious keywords") 
        if main_issue['normal_matches'] > 0:
            reason_parts.append(f"{main_issue['normal_matches']} normal keywords")
        if main_issue['position'] == 0:
            reason_parts.append("first question")
        
        reason = f"Score: {main_issue['score']} ({', '.join(reason_parts)})"
        
        return {
            'question': main_issue['question'],
            'question_time': main_issue['question_time'],
            'issue_type': issue_type,
            'confidence_score': min(main_issue['score'] / 10.0, 1.0),
            'all_candidates': scored_issues,
            'selected_reason': reason,
            'scoring_details': {
                'complaint_matches': main_issue['complaint_matches'],
                'serious_matches': main_issue['serious_matches'],
                'normal_matches': main_issue['normal_matches'],
                'is_initial_question': main_issue['is_initial_question'],
                'is_follow_up': main_issue['is_follow_up']
            }
        }

    def debug_scoring(self, qa_pairs):
        """Debug function untuk melihat detailed scoring"""
        if not qa_pairs:
            return
        
        main_issue = self.detect_main_issue(qa_pairs)
        
        print("🔍 DETAILED SCORING ANALYSIS:")
        print("=" * 60)
        
        for i, candidate in enumerate(main_issue['all_candidates']):
            print(f"\n{i+1}. '{candidate['question'][:60]}...'")
            print(f"   Score: {candidate['score']}")
            print(f"   Position: {candidate['position'] + 1}")
            print(f"   Word Count: {candidate['word_count']}")
            print(f"   Initial Question: {candidate['is_initial_question']}")
            print(f"   Follow-up: {candidate['is_follow_up']}")
            print(f"   Keywords: C:{candidate['complaint_matches']} S:{candidate['serious_matches']} N:{candidate['normal_matches']}")
            print(f"   Lead Time: {candidate['lead_time']} min")

# Hybrid ML Classifier
class HybridClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # REDUCE features
            ngram_range=(1, 1),  # Start with unigrams saja
            min_df=1,           # Less restrictive
            max_df=0.9,
            stop_words=None     # Jangan remove stop words, penting untuk context
        )
        self.classifier = LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=2000,      # More iterations
            class_weight='balanced',
            C=1.0              # Regularization
        )
        self.is_trained = False
        
        # Enhanced rule-based fallback
        self.rule_keywords = {
            'complaint': [
                'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan', 
                'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa', 'refund',
                'garansi ditolak', 'pelayanan buruk', 'tidak profesional', 'minta uang kembali'
            ],
            'serious': [
                'error', 'rusak', 'masalah', 'gagal', 'mogok', 'mati', 'tidak bisa', 
                'help', 'urgent', 'kendala', 'trouble', 'macet', 'hang', 'blank',
                'not responding', 'bermasalah', 'gangguan', 'starter', 'rem blong', 
                'overheating', 'transmisi', 'kelistrikan', 'aki soak', 'check engine'
            ],
            'normal': [
                'tanya', 'info', 'harga', 'berapa', 'cara', 'bagaimana', 'fungsi', 
                'promo', 'spesifikasi', 'fitur', 'mau tanya', 'boleh tanya', 
                'minta info', 'informasi', 'tanyakan', 'booking', 'test drive',
                'alamat', 'lokasi', 'jam operasional', 'servis', 'sparepart', 'dp'
            ]
        }
        
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.5
        
        print("✅ Hybrid Classifier initialized")
    
    def create_enhanced_training_data(self):
        """Create BETTER training data dengan lebih banyak samples dan balance"""
        enhanced_data = {
            'texts': [
                "mau tanya harga mobil avanza berapa?",
                "berapa harga toyota rush terbaru?",
                "info promo innova zenix terbaru",
                "bagaimana cara booking test drive?",
                "spesifikasi toyota fortuner lengkap",
                "alamat dealer terdekat di jakarta",
                "jam operasional bengkel toyota",
                "berapa biaya servis berkala avanza?",
                "cara aktivasi fitur t-intouch",
                "fungsi safety sense pada mobil",
                "minta info dp ringan kredit",
                "berapa lama waktu servis berkala?",
                "warna yang tersedia untuk calya",
                "beda innova zenix dan innova lama",
                "fasilitas di bengkel toyota",
                "syarat test drive mobil baru",
                "info asuransi mobil terbaik",
                "cara klaim garansi mobil",
                "berapa harga velg ori innova?",
                "lokasi dealer 24 jam",
                "info cicilan mobil tanpa dp",
                "berapa konsumsi bensin avanza?",
                "sparepart yang perlu diganti rutin",
                "cara perawatan mobil baru",
                "beda vvti dan d4d",
                "berapa harga oli mesin ori?",
                "jadwal service gratis pertama",
                "cara connect android auto",
                "info membership toyota",
                "berapa harga ban mobil avanza",
                
                "mobil saya error tidak bisa starter",
                "mesin bunyi aneh ada masalah serius",
                "aplikasi error terus tidak bisa login",
                "rem blong sangat berbahaya",
                "mogok di jalan butuh bantuan cepat",
                "mesin overheating terus menerus",
                "transmisi bermasalah tidak bisa pindah gigi",
                "kelistrikan error semua lampu mati",
                "aki soak tidak bisa starter pagi ini",
                "check engine menyala terus sejak kemarin",
                "mobil tiba-tiba mati di tol",
                "asap keluar dari kap mesin",
                "rem tidak berfungsi dengan baik",
                "setir berat sekali tidak bisa dibelokkan",
                "oli mesin bocor parah",
                "air radiator habis terus menerus",
                "mobil tidak bisa distarter sama sekali",
                "lampu dashboard berkedip semua",
                "ban pecah di jalan tol",
                "mesin bergetar sangat kencang",
                "kopling slip tidak bisa jalan",
                "ac tidak dingin sama sekali",
                "rem bunyi keras setiap kali diinjak",
                "mobil tidak bisa masuk gigi",
                "asap hitam keluar dari knalpot",
                "mesin sulit dinyakan di pagi hari",
                "rem tangan tidak bisa dilepas",
                "oli terus berkurang setiap hari",
                "mobil terbakar sendiri",
                "kaca spion patah karena kecelakaan",
                
                "saya komplain tentang pelayanan bengkel",
                "sangat kecewa dengan produk toyota ini",
                "komplain untuk garansi yang ditolak",
                "pelayanan customer service sangat buruk",
                "minta refund untuk produk cacat",
                "protes untuk biaya servis yang mahal",
                "pengaduan untuk teknisi tidak profesional",
                "kecewa dengan waiting time yang lama",
                "komplain sparepart palsu yang dipasang",
                "protes untuk janji tidak ditepati",
                "saya marah dengan kualitas servis",
                "komplain mobil baru langsung rusak",
                "kecewa dengan respon yang lambat",
                "minta ganti rugi untuk kerusakan",
                "protes harga sparepart terlalu mahal",
                "komplain untuk janji service tidak tepat waktu",
                "sangat tidak puas dengan pelayanan",
                "komplain mobil sering masuk bengkel",
                "kecewa dengan kualitas cat mobil",
                "protes untuk penanganan yang lamban",
                "komplain untuk informasi yang misleading",
                "saya keberatan dengan biaya tambahan",
                "komplain untuk attitude staff yang kasar",
                "kecewa dengan fitur yang tidak berfungsi",
                "protes untuk kebijakan yang tidak jelas",
                "komplain untuk janji telepon tidak ditepati",
                "sangat marah dengan pelayanan after sales",
                "komplain untuk sparepart tidak tersedia",
                "kecewa dengan waktu tunggu yang panjang",
                "protes untuk solusi yang tidak memuaskan"
            ],
            'labels': ['normal'] * 30 + ['serious'] * 30 + ['complaint'] * 30
        }
        
        return enhanced_data
    
    def train(self, X_train, y_train):
        """Train model dengan enhanced approach"""
        try:
            print("🤖 Training Enhanced ML model...")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Class distribution: {Counter(y_train)}")
            
            # Fit vectorizer dan transform data
            X_vec = self.vectorizer.fit_transform(X_train)
            
            # Train classifier
            self.classifier.fit(X_vec, y_train)
            self.is_trained = True
            
            # Calculate training accuracy
            train_pred = self.classifier.predict(X_vec)
            train_accuracy = accuracy_score(y_train, train_pred)
            
            # Save model
            joblib.dump(self.vectorizer, f"{config.MODEL_SAVE_PATH}/vectorizer.pkl")
            joblib.dump(self.classifier, f"{config.MODEL_SAVE_PATH}/classifier.pkl")
            
            print(f"✅ Model trained - Accuracy: {train_accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def predict(self, text):
        """Predict dengan hybrid approach"""
        if not text or len(text.strip()) < 3:
            return 'normal', 0.5
        
        text_lower = text.lower()
        
        # Step 1: Try ML prediction jika model sudah trained
        ml_prediction, ml_confidence = None, 0.0
        if self.is_trained:
            try:
                X_vec = self.vectorizer.transform([text])
                ml_prediction = self.classifier.predict(X_vec)[0]
                ml_probs = self.classifier.predict_proba(X_vec)[0]
                ml_confidence = np.max(ml_probs)
                
                # High confidence ML prediction
                if ml_confidence >= self.high_confidence_threshold:
                    return ml_prediction, ml_confidence
                    
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
        
        # Step 2: Rule-based prediction
        rule_prediction = self._rule_based_predict(text_lower)
        rule_confidence = self._calculate_rule_confidence(text_lower, rule_prediction)
        
        # Step 3: Hybrid decision making
        if self.is_trained and ml_prediction:
            if ml_confidence >= self.medium_confidence_threshold:
                if ml_prediction == rule_prediction:
                    hybrid_confidence = (ml_confidence + rule_confidence) / 2
                    return ml_prediction, hybrid_confidence
                else:
                    # Conflict - prefer rules untuk safety
                    return rule_prediction, rule_confidence * 0.8
            else:
                return rule_prediction, rule_confidence
        else:
            return rule_prediction, rule_confidence
    
    def _rule_based_predict(self, text_lower):
        """Rule-based classification fallback"""
        scores = {'normal': 0, 'serious': 0, 'complaint': 0}
        
        for category, keywords in self.rule_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == 'complaint':
                        scores[category] += 3
                    elif category == 'serious':
                        scores[category] += 2
                    else:
                        scores[category] += 1
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'normal'
        
        max_categories = [cat for cat, score in scores.items() if score == max_score]
        if len(max_categories) > 1:
            if 'serious' in max_categories:
                return 'serious'
            elif 'complaint' in max_categories:
                return 'complaint'
            else:
                return 'normal'
        
        return max_categories[0]
    
    def _calculate_rule_confidence(self, text_lower, predicted_category):
        """Calculate confidence score untuk rule-based prediction"""
        scores = {'normal': 0, 'serious': 0, 'complaint': 0}
        total_weight = 0
        
        for category, keywords in self.rule_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    weight = 3 if category == 'complaint' else (2 if category == 'serious' else 1)
                    scores[category] += weight
                    total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        max_score = max(scores.values())
        confidence = max_score / total_weight
        
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 2:
            confidence = min(confidence * 1.2, 0.95)
        
        return confidence
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance dengan detailed report"""
        if not self.is_trained:
            print("❌ Model not trained")
            return None
        
        X_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_vec)
        y_pred_proba = self.classifier.predict_proba(X_vec)
        
        print("📊 ENHANCED CLASSIFIER PERFORMANCE REPORT")
        print("=" * 50)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Test Samples: {len(X_test)}")
        print(f"Class Distribution: {Counter(y_test)}")
        
        print("\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['normal', 'serious', 'complaint']))
        
        max_confidences = np.max(y_pred_proba, axis=1)
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Average Confidence: {np.mean(max_confidences):.3f}")
        print(f"   High Confidence (>0.7): {np.sum(max_confidences > 0.7)}/{len(max_confidences)}")
        print(f"   Low Confidence (<0.5): {np.sum(max_confidences < 0.5)}/{len(max_confidences)}")
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['normal', 'serious', 'complaint'],
                   yticklabels=['normal', 'serious', 'complaint'])
        plt.title('Confusion Matrix - Enhanced Classifier')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
        
        # Feature importance
        self._show_feature_importance()
        
        return accuracy
    
    def _show_feature_importance(self, top_n=15):
        """Show most important features untuk setiap class"""
        if not self.is_trained:
            return
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"\n🔍 Top {top_n} Features per Class:")
        
        for i, class_name in enumerate(['normal', 'serious', 'complaint']):
            coef = self.classifier.coef_[i]
            top_indices = np.argsort(coef)[-top_n:][::-1]
            top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]
            
            print(f"\n{class_name.upper()}:")
            for feature, score in top_features:
                print(f"   {feature}: {score:.3f}")

class ReplyAnalyzer:
    def __init__(self):
        # First reply patterns
        self.first_reply_patterns = [
            r'tangkapan', r'layar', r'cek', r'proses', r'kami\s+lihat', r'kami\s+periksa', 
            r'konfirmasi', r'validasi', r'follow\s+up', r'tindak\s+lanjut', r'eskalasi',
            r'kami\s+pelajari', r'kami\s+investigasi', r'kami\s+telusuri', r'disarankan',
            r'jika\s+dilihat', r'dilihat\s+dari', r'berdasarkan.*layar', r'foto.*yang.*kirim',
            r'screenshot.*yang', r'gambar.*yang.*kirim', r'dari.*foto', r'akan\s+kami',
            r'pengecekan', r'verifikasi', r'kami\s+teruskan', r'kami\s+diskusikan',
            r'tunggu\s+sebentar', r'mohon\s+ditunggu', r'proses', r'cek\s+dulu'
        ]

        # Final reply patterns
        self.solution_patterns = [
            r'solusi', r'jawaban', r'caranya', r'prosedur', r'bisa\s+menghubungi',
            r'silakan\s+menghubungi', r'disarankan', r'disarankan\s+untuk', r'rekomendasi',
            r'berikut\s+informasi', r'nomor\s+telepon', r'alamat\s+dealer', r'bengkel\s+resmi',
            r'call\s+center', r'hotline', r'customer\s+service', r'info\s+lengkap',
            r'cara\s+mengaktifkan', r'langkah-langkah', r'penjelasan\s+tentang', r'harga\s+mulai',
            r'biaya\s+required', r'tarif\s+berlaku', r'jam\s+operasional', r'alamat\s+lengkap',
            r'syarat\s+dan\s+ketentuan', r'spesifikasi', r'fungsi', r'bisa\s+dilakukan',
            r'dapat\s+dilakukan', r'silakan\s+datang', r'bawa\s+ke', r'perbaikan', r'servis',
            r'ganti', r'penyebabnya', r'akibat', r'rekomendasi\s+kami', r'bisa\s+dicoba\s+kembali'
        ]
        
        # Escalation patterns
        self.escalation_patterns = [
            r'akan\s+ditindaklanjuti', r'diteruskan\s+ke', r'akan\s+diteruskan', 
            r'dilaporkan\s+ke', r'akan\s+dilaporkan', r'tunggu\s+informasi', 
            r'follow\s+up', r'proses\s+lebih\s+lanjut', r'akan\s+kami\s+proses',
            r'dibantu\s+teruskan', r'disampaikan\s+lebih\s+lanjut', r'pihak\s+terkait',
            r'tim\s+terkait', r'area\s+terkait', r'akan\s+diperbaiki', r'akan\s+dicek'
        ]
        
        # Conversation enders (TAMBAHAN: skip yang cuma nanya "apakah sudah jelas")
        self.conversation_ender_patterns = [
            r'terima\s+kasih', r'thanks', r'makasih', r'tks', r'sampai\s+jumpa', r'semoga\s+membantu',
            r'goodbye', r'bye', r'dadah', r'live\s+chat\s+ditutup', r'chat\s+saya\s+tutup',
            r'jika\s+tidak\s+ada\s+hal\s+lain', r'jika\s+ada\s+pertanyaan\s+lagi'
        ]
        
        # SKIP patterns - reply yang harus di-skip karena tidak meaningful (TAMBAHAN BARU)
        self.skip_patterns = [
            r'apakah\s+informasinya\s+sudah\s+cukup\s+jelas',
            r'apakah\s+sudah\s+cukup\s+jelas', 
            r'apakah\s+jelas',
            r'sudah\s+cukup\s+jelas',
            r'cukup\s+jelas',
            r'apakah\s+ada\s+hal\s+lain',
            r'ada\s+hal\s+lain',
            r'apakah\s+ada\s+pertanyaan\s+lain',
            r'ada\s+pertanyaan\s+lain',
            r'apakah\s+bisa\s+dibantu',
            r'bisa\s+dibantu\s+lagi',
            r'mau\s+tanya\s+lagi',
            r'ingin\s+tanya\s+lagi'
        ]
        
        # Generic/bot replies
        self.generic_reply_patterns = [
            r'virtual\s+assistant', r'akan\s+segera\s+menghubungi', r'dalam\s+antrian',
            r'terima\s+kasih,\s+saat\s+ini\s+anda\s+masuk', r'customer\s+service\s+akan',
            r'menghubungi\s+anda', r'silakan\s+memilih\s+dari\s+menu', r'klik\s+setuju',
            r'data\s+privasi', r'pilih\s+menu', r'silahkan\s+ketik\s+nama'
        ]

    def analyze_replies(self, qa_pairs, main_issue_type, customer_info=None, all_tickets_data=None):
        """Analyze replies dengan SKIP patterns baru"""
        if not qa_pairs:
            return None, None, self._create_empty_analysis(main_issue_type, "No Q-A pairs available")
        
        print(f"🔍 Analyzing {len(qa_pairs)} replies for {main_issue_type} issue...")
        
        # Check customer leave condition
        customer_leave = self._detect_customer_leave(qa_pairs)
        
        # Cari first reply (skip yang meaningless)
        first_reply = self._find_first_reply(qa_pairs)
        
        # Cari final reply (skip yang meaningless)  
        final_reply = self._find_final_reply(qa_pairs, main_issue_type, customer_leave, first_reply)
        
        # 🔥 LOGIC: Jika customer leave, gunakan last meaningful operator message
        if customer_leave and not final_reply:
            last_operator_reply = self._get_last_meaningful_operator_reply(qa_pairs)
            if last_operator_reply:
                print("   🚶 Using last operator reply as final reply (customer leave)")
                final_reply = self._create_reply_object(last_operator_reply, 'final_customer_leave')
                final_reply['note'] = "Customer left conversation"
        
        # 🔥 LOGIC: Untuk escalation cases, first reply = escalation reply
        escalation_reply = self._find_escalation_reply(qa_pairs)
        if escalation_reply and main_issue_type in ['serious', 'complaint']:
            print("   🔄 Escalation case detected - using escalation as first reply")
            first_reply = escalation_reply
        
        # Validasi requirements
        is_valid = self._validate_requirements_simple(first_reply, final_reply, main_issue_type, customer_leave)
        
        # Calculate lead times
        lead_times = self._calculate_lead_times(qa_pairs, first_reply, final_reply, main_issue_type)
        
        # Build analysis result
        analysis_result = {
            'issue_type': main_issue_type,
            'first_reply': first_reply,
            'final_reply': final_reply,
            'lead_times': lead_times,
            'customer_leave': customer_leave,
            'escalation_case': escalation_reply is not None,
            'reply_validation': self.__validate_replies_simple(first_reply, final_reply, main_issue_type, customer_leave),
            'requirement_compliant': is_valid
        }
        
        print(f"   ✅ Reply analysis completed - Customer Leave: {customer_leave}, Escalation: {escalation_reply is not None}")
        return first_reply, final_reply, analysis_result

    def _find_final_reply(self, qa_pairs, issue_type, customer_leave, first_reply):
        """Cari final reply yang MEANINGFUL (skip yang meaningless)"""
        # Cari dari akhir ke awal
        for pair in reversed(qa_pairs):
            if pair['is_answered']:
                answer = pair['answer']
                
                # 🔥 SKIP jika meaningless/questioning reply
                if self._is_skip_reply(answer):
                    print(f"   ⏭️  Skipping meaningless reply: {answer[:50]}...")
                    continue
                
                # Untuk normal issues, prioritaskan solution reply
                if issue_type == 'normal' and self._is_solution_reply(answer):
                    print(f"   ✅ Final reply (solution): {answer[:60]}...")
                    return self._create_reply_object(pair, 'final_solution')
                
                # Untuk semua issues, cari meaningful conversation ender
                if self._is_meaningful_conversation_ender(answer):
                    print(f"   ✅ Final reply (meaningful ender): {answer[:60]}...")
                    return self._create_reply_object(pair, 'final_conversation_ender')
        
        # Fallback: gunakan last meaningful operator reply
        last_operator_reply = self._get_last_meaningful_operator_reply(qa_pairs)
        if last_operator_reply:
            print(f"   🔄 Final reply (last meaningful): {last_operator_reply['answer'][:60]}...")
            return self._create_reply_object(last_operator_reply, 'final_last_operator')
        
        return None

    def _find_first_reply(self, qa_pairs):
        """Cari first reply yang MEANINGFUL (skip yang meaningless)"""
        for pair in qa_pairs:
            if pair['is_answered']:
                answer = pair['answer']
                
                # 🔥 SKIP jika meaningless/generic reply
                if self._is_skip_reply(answer) or self._is_generic_reply(answer):
                    print(f"   ⏭️  Skipping meaningless first reply: {answer[:50]}...")
                    continue
                
                # Ambil first meaningful reply
                print(f"   ✅ First reply found: {answer[:60]}...")
                return self._create_reply_object(pair, 'first_standard')
        
        return None

    def _find_escalation_reply(self, qa_pairs):
        """Cari escalation reply yang MEANINGFUL"""
        for pair in qa_pairs:
            if pair['is_answered'] and self._is_escalation_reply(pair['answer']):
                # Pastikan bukan meaningless reply
                if not self._is_skip_reply(pair['answer']):
                    print(f"   🔄 Escalation reply found: {pair['answer'][:60]}...")
                    return self._create_reply_object(pair, 'first_escalation')
        return None

    def _get_last_meaningful_operator_reply(self, qa_pairs):
        """Ambil operator reply terakhir yang MEANINGFUL"""
        operator_replies = []
        for pair in reversed(qa_pairs):  # Cari dari akhir
            if pair['is_answered']:
                role = pair.get('answer_role', '').lower()
                if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                    answer = pair['answer']
                    # 🔥 SKIP jika meaningless reply
                    if not self._is_skip_reply(answer) and not self._is_generic_reply(answer):
                        operator_replies.append(pair)
        
        if operator_replies:
            return operator_replies[0]  # Yang terakhir (karena sudah reversed)
        return None

    def _is_skip_reply(self, message):
        """Cek apakah reply harus di-SKIP karena meaningless/questioning"""
        message_lower = message.lower()
        
        # Skip jika hanya menanyakan "apakah sudah jelas" dll
        if any(re.search(pattern, message_lower) for pattern in self.skip_patterns):
            return True
        
        # Skip jika sangat pendek dan tidak meaningful
        words = message_lower.split()
        if len(words) <= 3:
            questioning_words = ['apakah', 'apakah', 'sudah', 'cukup', 'jelas', 'ada', 'hal', 'lain', 'bisa', 'dibantu']
            if any(word in words for word in questioning_words):
                return True
        
        return False

    def _is_meaningful_conversation_ender(self, message):
        """Cek apakah meaningful conversation ender (bukan cuma nanya 'apakah jelas')"""
        message_lower = message.lower()
        
        # Skip jika hanya menanyakan tanpa memberikan value
        if self._is_skip_reply(message):
            return False
        
        # Cek jika mengandung conversation ender patterns DAN meaningful content
        has_ender = any(re.search(pattern, message_lower) for pattern in self.conversation_ender_patterns)
        has_meaningful_content = len(message_lower.split()) > 5  # Minimal 5 kata
        
        return has_ender and has_meaningful_content

    def _is_solution_reply(self, message):
        """Cek apakah message mengandung solusi"""
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in self.solution_patterns)

    def _is_escalation_reply(self, message):
        """Cek apakah message mengandung eskalasi"""
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in self.escalation_patterns)

    def _is_generic_reply(self, message):
        """Skip generic/bot replies"""
        message_lower = str(message).lower()
        return any(pattern in message_lower for pattern in self.generic_reply_patterns)

    def _detect_customer_leave(self, qa_pairs):
        """Deteksi customer leave"""
        if not qa_pairs:
            return False
        
        # Cek jika ada unanswered questions di akhir
        unanswered_count = sum(1 for pair in qa_pairs[-3:] if not pair['is_answered'])
        if unanswered_count >= 2:
            print("   🚶 Customer leave detected: multiple unanswered questions at end")
            return True
        
        return False

    def _validate_requirements_simple(self, first_reply, final_reply, issue_type, customer_leave):
        """Validasi requirements sederhana"""
        if customer_leave:
            return True  # Relaxed requirements untuk customer leave
        
        if issue_type == 'normal':
            return final_reply is not None
        
        if issue_type in ['serious', 'complaint']:
            return first_reply is not None
        
        return True

    def _validate_replies_simple(self, first_reply, final_reply, issue_type, customer_leave):
        """Validasi replies sederhana"""
        validation = {
            'first_reply_found': first_reply is not None,
            'final_reply_found': final_reply is not None,
            'customer_leave': customer_leave,
            'quality_score': 0,
            'quality_rating': 'fair'
        }
        
        if first_reply:
            validation['quality_score'] += 2
        if final_reply:
            validation['quality_score'] += 2
        
        if customer_leave:
            validation['recommendation'] = 'Customer left conversation'
            validation['quality_rating'] = 'fair'
        elif not final_reply and issue_type == 'normal':
            validation['recommendation'] = 'Final reply missing'
            validation['quality_rating'] = 'poor'
        elif not first_reply and issue_type in ['serious', 'complaint']:
            validation['recommendation'] = 'First reply missing'
            validation['quality_rating'] = 'poor'
        else:
            validation['recommendation'] = 'Requirements met'
            validation['quality_rating'] = 'good'
        
        return validation

    def _create_reply_object(self, pair, reply_type):
        """Create standardized reply object"""
        return {
            'message': pair['answer'],
            'timestamp': pair['answer_time'],
            'role': pair['answer_role'],
            'reply_type': reply_type,
            'lead_time_seconds': pair.get('lead_time_seconds'),
            'lead_time_minutes': pair.get('lead_time_minutes'),
            'lead_time_hhmmss': pair.get('lead_time_hhmmss'),
            'question': pair['question'],
            'question_time': pair['question_time']
        }

    def _calculate_lead_times(self, qa_pairs, first_reply, final_reply, issue_type):
        """Hitung lead times"""
        lead_times = {}
        
        if not qa_pairs:
            return lead_times
        
        main_question_time = qa_pairs[0]['question_time']
        
        if first_reply and first_reply['timestamp']:
            lead_times['first_reply_lead_time_seconds'] = (
                first_reply['timestamp'] - main_question_time
            ).total_seconds()
            lead_times['first_reply_lead_time_minutes'] = round(
                lead_times['first_reply_lead_time_seconds'] / 60, 2
            )
        
        if final_reply and final_reply['timestamp']:
            lead_times['final_reply_lead_time_seconds'] = (
                final_reply['timestamp'] - main_question_time
            ).total_seconds()
            lead_times['final_reply_lead_time_minutes'] = round(
                lead_times['final_reply_lead_time_seconds'] / 60, 2
            )
        
        return lead_times

    def _create_empty_analysis(self, issue_type, reason):
        """Create empty analysis result"""
        return {
            'issue_type': issue_type,
            'first_reply': None,
            'final_reply': None,
            'lead_times': {},
            'reply_validation': {
                'first_reply_found': False,
                'final_reply_found': False,
                'recommendation': reason,
                'quality_score': 0,
                'quality_rating': 'failed'
            },
            'requirement_compliant': False
        }
        
# CompleteAnalysisPipeline (FULL FIX)
import time
from datetime import datetime
from collections import Counter

class CompleteAnalysisPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.parser = ConversationParser()
        self.issue_detector = MainIssueDetector()
        self.classifier = HybridClassifier()
        self.reply_analyzer = ReplyAnalyzer()  # Gunakan yang baru
        self.results = []
        self.analysis_stats = {}
        self.customer_info = {}
        self.all_tickets_qa_pairs = {}
        
        print("🚀 Complete Analysis Pipeline Initialized")
        print("   ✓ Data Preprocessor")
        print("   ✓ Conversation Parser") 
        print("   ✓ Main Issue Detector")
        print("   ✓ Hybrid Classifier")
        print("   ✓ Reply Analyzer")
    
    def analyze_single_ticket(self, ticket_df, ticket_id, all_tickets_data=None):
        """Analisis lengkap untuk single ticket - DENGAN LOGIC BARU"""
        print(f"🎯 Analyzing Ticket: {ticket_id}")
        
        try:
            # Step 1: Parse Q-A pairs
            qa_pairs = self.parser.parse_conversation(ticket_df)
            
            # Simpan Q-A pairs untuk follow-up analysis
            self.all_tickets_qa_pairs[ticket_id] = qa_pairs
            
            if not qa_pairs:
                return self._create_ticket_result(ticket_id, "failed", "No Q-A pairs detected", {})
            
            answered_count = sum(1 for pair in qa_pairs if pair['is_answered'])
            print(f"   ✓ Found {len(qa_pairs)} Q-A pairs ({answered_count} answered)")
            
            # Step 2: Detect main issue
            main_issue = self.issue_detector.detect_main_issue(qa_pairs)
            
            if not main_issue:
                return self._create_ticket_result(ticket_id, "failed", "No main issue detected", {})
            
            print(f"   ✓ Main issue: {main_issue['issue_type']} (conf: {main_issue['confidence_score']:.2f})")
            
            # Step 3: Classify issue type
            ml_prediction, ml_confidence = self.classifier.predict(main_issue['question'])
            final_issue_type = self._resolve_issue_type(main_issue['issue_type'], ml_prediction, ml_confidence)
            
            print(f"   ✓ Final classification: {final_issue_type} (ML conf: {ml_confidence:.2f})")
            
            # Step 4: Analyze replies dengan LOGIC BARU
            first_reply, final_reply, reply_analysis = self.reply_analyzer.analyze_replies(
                qa_pairs, final_issue_type, self.customer_info, self.all_tickets_qa_pairs
            )
            
            # Step 5: Compile results
            result = self._compile_ticket_result(
                ticket_id, ticket_df, qa_pairs, main_issue, final_issue_type,
                ml_prediction, ml_confidence, first_reply, final_reply, reply_analysis
            )
            
            print(f"   ✅ Analysis completed - Performance: {result['performance_rating'].upper()}")
            return result
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"   ❌ Analysis failed: {error_msg}")
            return self._create_ticket_result(ticket_id, "failed", error_msg, {})

    def _compile_ticket_result(self, ticket_id, ticket_df, qa_pairs, main_issue, final_issue_type,
                             ml_prediction, ml_confidence, first_reply, final_reply, reply_analysis):
        """Compile ticket result dengan logic baru"""
        
        # Tentukan performance rating berdasarkan reply analysis
        if reply_analysis['requirement_compliant']:
            performance_rating = 'good'
        else:
            performance_rating = 'fair'
        
        # Hitung quality score
        quality_score = 0
        if first_reply:
            quality_score += 2
        if final_reply:
            quality_score += 2
        if reply_analysis.get('customer_leave'):
            quality_score += 1  # Bonus point untuk deteksi customer leave
        
        return {
            'ticket_id': ticket_id,
            'status': 'success',
            'analysis_timestamp': datetime.now(),
            
            # Conversation info
            'total_messages': len(ticket_df),
            'total_qa_pairs': len(qa_pairs),
            'answered_pairs': sum(1 for pair in qa_pairs if pair['is_answered']),
            'customer_leave': reply_analysis.get('customer_leave', False),
            
            # Main issue analysis
            'main_question': main_issue['question'],
            'main_question_time': main_issue['question_time'],
            'detected_issue_type': main_issue['issue_type'],
            'final_issue_type': final_issue_type,
            'detection_confidence': main_issue['confidence_score'],
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            
            # Reply analysis - LOGIC BARU
            'first_reply_found': first_reply is not None,
            'final_reply_found': final_reply is not None,
            'first_reply_message': first_reply['message'] if first_reply else None,
            'first_reply_time': first_reply['timestamp'] if first_reply else None,
            'final_reply_message': final_reply['message'] if final_reply else None,
            'final_reply_time': final_reply['timestamp'] if final_reply else None,
            'customer_leave_note': final_reply.get('customer_leave_note') if final_reply else None,
            'follow_up_ticket': final_reply.get('follow_up_ticket') if final_reply else None,
            
            # Lead times
            'first_reply_lead_time_minutes': reply_analysis['lead_times'].get('first_reply_lead_time_minutes'),
            'final_reply_lead_time_minutes': reply_analysis['lead_times'].get('final_reply_lead_time_minutes'),
            'first_reply_lead_time_hhmmss': reply_analysis['lead_times'].get('first_reply_lead_time_hhmmss'),
            'final_reply_lead_time_hhmmss': reply_analysis['lead_times'].get('final_reply_lead_time_hhmmss'),
            
            # Performance metrics
            'performance_rating': performance_rating,
            'quality_score': quality_score,
            'quality_rating': reply_analysis['reply_validation']['quality_rating'],
            
            # Recommendations
            'recommendation': reply_analysis['reply_validation']['recommendation'],
            'requirement_compliant': reply_analysis['requirement_compliant'],
            
            # Raw data untuk export
            '_raw_qa_pairs': qa_pairs,
            '_raw_main_issue': main_issue,
            '_raw_reply_analysis': reply_analysis
        }

    def _calculate_conversation_duration(self, ticket_df):
        """Hitung durasi conversation dalam menit"""
        if len(ticket_df) < 2:
            return 0
        
        try:
            start_time = ticket_df['parsed_timestamp'].min()
            end_time = ticket_df['parsed_timestamp'].max()
            duration_minutes = (end_time - start_time).total_seconds() / 60
            return round(duration_minutes, 2)
        except:
            return 0
    
    def _resolve_issue_type(self, detected_type, ml_prediction, ml_confidence):
        """Resolve final issue type antara rule-based dan ML"""
        if ml_confidence > 0.7:  # High confidence ML
            return ml_prediction
        elif ml_confidence > 0.5 and ml_prediction == detected_type:  # Consistent
            return ml_prediction
        else:  # Trust rule-based detection
            return detected_type
    
    def _extract_threshold_violations(self, threshold_checks):
        """Extract threshold violations dari analysis"""
        violations = []
        for key, value in threshold_checks.items():
            if 'exceeded' in key and value:
                violations.append(key)
        return violations
    
    def _create_ticket_result(self, ticket_id, status, reason, extra_data):
        """Create standardized result object"""
        result = {
            'ticket_id': ticket_id,
            'status': status,
            'failure_reason': reason if status == 'failed' else None,
            'analysis_timestamp': datetime.now()
        }
        result.update(extra_data)
        return result
    
    def analyze_all_tickets(self, df, sample_size=None, max_tickets=None):
        """Analisis semua tickets dengan comprehensive reporting"""
        print("🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        print("=" * 60)
        
        # Extract customer info terlebih dahulu
        print("📋 Extracting customer information...")
        self.customer_info = self.preprocessor.extract_customer_info(df)
        print(f"   ✅ Extracted info for {len(self.customer_info)} tickets")
        
        ticket_ids = df['Ticket Number'].unique()
        
        if sample_size:
            ticket_ids = ticket_ids[:sample_size]
            print(f"🔍 Analyzing {sample_size} sample tickets...")
        elif max_tickets:
            ticket_ids = ticket_ids[:max_tickets]
            print(f"🔍 Analyzing {max_tickets} tickets (max limit)...")
        else:
            print(f"🔍 Analyzing {len(ticket_ids)} tickets...")
        
        self.results = []
        successful_analyses = 0
        
        start_time = time.time()
        
        for i, ticket_id in enumerate(ticket_ids):
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            result = self.analyze_single_ticket(ticket_df, ticket_id, self.all_tickets_qa_pairs)
            self.results.append(result)
            
            if result['status'] == 'success':
                successful_analyses += 1
            
            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(ticket_ids):
                progress = (i + 1) / len(ticket_ids) * 100
                print(f"   📊 Progress: {i + 1}/{len(ticket_ids)} ({progress:.1f}%) - {successful_analyses} successful")
        
        # Calculate comprehensive statistics
        analysis_time = time.time() - start_time
        self.analysis_stats = self._calculate_comprehensive_stats(analysis_time, len(ticket_ids))
        
        print(f"\n🎉 ANALYSIS PIPELINE COMPLETED!")
        print("=" * 60)
        self._print_summary_report()
        
        return self.results, self.analysis_stats
    
    def _calculate_comprehensive_stats(self, analysis_time, total_tickets):
        """Hitung comprehensive statistics dari results - FIXED VERSION"""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        # BASIC STATS - PASTIKAN SEMUA KEY ADA
        stats = {
            'total_tickets': len(self.results),
            'successful_analysis': len(successful),
            'failed_analysis': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'analysis_duration_seconds': analysis_time,
            'avg_analysis_time_per_ticket': analysis_time / len(self.results) if self.results else 0
        }
        
        if not successful:
            return stats
        
        # Issue type distribution
        issue_types = [r['final_issue_type'] for r in successful]
        stats['issue_type_distribution'] = dict(Counter(issue_types))
        
        # Performance metrics
        performance_ratings = [r['performance_rating'] for r in successful]
        stats['performance_distribution'] = dict(Counter(performance_ratings))
        
        # Lead time statistics - PERBAIKAN: Hitung average lead time per issue type
        lead_time_stats = self._calculate_lead_time_statistics(successful)
        stats.update(lead_time_stats)
        
        # Quality metrics
        quality_scores = [r['quality_score'] for r in successful]
        stats['quality_stats'] = {
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'avg_quality_rating': Counter([r['quality_rating'] for r in successful]).most_common(1)[0][0] if successful else 'N/A'
        }
        
        # Threshold violations
        all_violations = []
        for r in successful:
            all_violations.extend(r.get('threshold_violations', []))
        stats['threshold_violations'] = dict(Counter(all_violations))
        
        # Reply effectiveness
        stats['reply_effectiveness'] = {
            'first_reply_found_rate': sum(1 for r in successful if r['first_reply_found']) / len(successful) if successful else 0,
            'final_reply_found_rate': sum(1 for r in successful if r['final_reply_found']) / len(successful) if successful else 0,
            'both_replies_found_rate': sum(1 for r in successful if r['first_reply_found'] and r['final_reply_found']) / len(successful) if successful else 0,
            'customer_leave_cases': sum(1 for r in successful if r.get('customer_leave', False)),
            'follow_up_cases': sum(1 for r in successful if r.get('follow_up_ticket'))
        }

        # Q-A Pairs Statistics
        total_qa_pairs = sum(r['total_qa_pairs'] for r in successful)
        total_answered_pairs = sum(r['answered_pairs'] for r in successful)
        stats['qa_pairs_stats'] = {
            'total_qa_pairs': total_qa_pairs,
            'total_answered_pairs': total_answered_pairs,
            'answer_rate': total_answered_pairs / total_qa_pairs if total_qa_pairs > 0 else 0,
            'avg_qa_pairs_per_ticket': total_qa_pairs / len(successful) if successful else 0
        }

        # Raw Data Availability
        raw_data_available = {
            'qa_pairs_available': sum(1 for r in successful if '_raw_qa_pairs' in r),
            'main_issue_available': sum(1 for r in successful if '_raw_main_issue' in r),
            'reply_analysis_available': sum(1 for r in successful if '_raw_reply_analysis' in r)
        }
        stats['raw_data_availability'] = raw_data_available
        
        return stats

    def _calculate_lead_time_statistics(self, successful_results):
        """Hitung statistik lead time keseluruhan dan per issue type"""
        lead_time_stats = {}
        
        # Lead time keseluruhan
        all_first_lead_times = [r['first_reply_lead_time_minutes'] for r in successful_results 
                               if r.get('first_reply_lead_time_minutes') is not None]
        all_final_lead_times = [r['final_reply_lead_time_minutes'] for r in successful_results 
                               if r.get('final_reply_lead_time_minutes') is not None]
        
        # Overall statistics - PASTIKAN DEFAULT VALUES
        lead_time_stats['overall_lead_times'] = {
            'first_reply_avg_minutes': np.mean(all_first_lead_times) if all_first_lead_times else 0,
            'first_reply_median_minutes': np.median(all_first_lead_times) if all_first_lead_times else 0,
            'final_reply_avg_minutes': np.mean(all_final_lead_times) if all_final_lead_times else 0,
            'final_reply_median_minutes': np.median(all_final_lead_times) if all_final_lead_times else 0,
            'first_reply_samples': len(all_first_lead_times),
            'final_reply_samples': len(all_final_lead_times)
        }
        
        # Per issue type statistics
        issue_type_lead_times = {}
        
        for result in successful_results:
            issue_type = result['final_issue_type']
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            if issue_type not in issue_type_lead_times:
                issue_type_lead_times[issue_type] = {
                    'first_reply_times': [],
                    'final_reply_times': []
                }
            
            if first_lt is not None:
                issue_type_lead_times[issue_type]['first_reply_times'].append(first_lt)
            if final_lt is not None:
                issue_type_lead_times[issue_type]['final_reply_times'].append(final_lt)
        
        # Calculate averages per issue type
        lead_time_stats['issue_type_lead_times'] = {}
        
        for issue_type, times in issue_type_lead_times.items():
            first_avg = np.mean(times['first_reply_times']) if times['first_reply_times'] else 0
            final_avg = np.mean(times['final_reply_times']) if times['final_reply_times'] else 0
            
            lead_time_stats['issue_type_lead_times'][issue_type] = {
                'first_reply_avg_minutes': first_avg,
                'final_reply_avg_minutes': final_avg,
                'first_reply_samples': len(times['first_reply_times']),
                'final_reply_samples': len(times['final_reply_times'])
            }
        
        return lead_time_stats
    
    def _print_summary_report(self):
        """Print comprehensive summary report dengan lead time statistics - FIXED VERSION"""
        stats = self.analysis_stats
        
        print(f"📊 COMPREHENSIVE ANALYSIS REPORT")
        print(f"   • Total Tickets: {stats.get('total_tickets', 0)}")
        print(f"   • Successful Analysis: {stats.get('successful_analysis', 0)} ({stats.get('success_rate', 0)*100:.1f}%)")
        print(f"   • Analysis Duration: {stats.get('analysis_duration_seconds', 0):.1f}s")
        print(f"   • Avg Time per Ticket: {stats.get('avg_analysis_time_per_ticket', 0):.2f}s")
        
        # LEAD TIME SUMMARY - REQUIREMENT BARU
        if 'overall_lead_times' in stats:
            overall_lt = stats['overall_lead_times']
            print(f"\n⏱️ OVERALL LEAD TIME SUMMARY:")
            print(f"   • First Reply - Avg: {overall_lt.get('first_reply_avg_minutes', 0):.1f} min (n={overall_lt.get('first_reply_samples', 0)})")
            print(f"   • Final Reply - Avg: {overall_lt.get('final_reply_avg_minutes', 0):.1f} min (n={overall_lt.get('final_reply_samples', 0)})")
        
        # LEAD TIME PER ISSUE TYPE - REQUIREMENT BARU
        if 'issue_type_lead_times' in stats:
            print(f"\n📈 LEAD TIME BY ISSUE TYPE:")
            for issue_type, lt_stats in stats['issue_type_lead_times'].items():
                first_avg = lt_stats.get('first_reply_avg_minutes', 0)
                final_avg = lt_stats.get('final_reply_avg_minutes', 0)
                
                print(f"   • {issue_type.upper()}:")
                print(f"     - First Reply: {first_avg:.1f} min (n={lt_stats.get('first_reply_samples', 0)})")
                print(f"     - Final Reply: {final_avg:.1f} min (n={lt_stats.get('final_reply_samples', 0)})")
        
        if 'issue_type_distribution' in stats:
            print(f"\n🎯 ISSUE TYPE DISTRIBUTION:")
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                print(f"   • {issue_type.title()}: {count} ({percentage:.1f}%)")
        
        if 'performance_distribution' in stats:
            print(f"\n📈 PERFORMANCE DISTRIBUTION:")
            for rating, count in stats['performance_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                print(f"   • {rating.upper()}: {count} ({percentage:.1f}%)")
        
        if 'threshold_violations' in stats:
            print(f"\n🚨 THRESHOLD VIOLATIONS:")
            for violation, count in stats['threshold_violations'].items():
                print(f"   • {violation}: {count}")
        
        if 'reply_effectiveness' in stats:
            eff = stats['reply_effectiveness']
            print(f"\n💬 REPLY EFFECTIVENESS:")
            print(f"   • First Reply Found: {eff.get('first_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Final Reply Found: {eff.get('final_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Both Replies Found: {eff.get('both_replies_found_rate', 0)*100:.1f}%")
            print(f"   • Customer Leave Cases: {eff.get('customer_leave_cases', 0)}")
            print(f"   • Follow-up Cases: {eff.get('follow_up_cases', 0)}")

        if 'qa_pairs_stats' in stats:
            qa_stats = stats['qa_pairs_stats']
            print(f"\n🔗 Q-A PAIRS STATISTICS:")
            print(f"   • Total Q-A Pairs: {qa_stats.get('total_qa_pairs', 0)}")
            print(f"   • Answered Pairs: {qa_stats.get('total_answered_pairs', 0)} ({qa_stats.get('answer_rate', 0)*100:.1f}%)")
            print(f"   • Avg Pairs per Ticket: {qa_stats.get('avg_qa_pairs_per_ticket', 0):.1f}")

    def export_results(self, output_file="output/pipeline_results.xlsx"):
        """Export results ke Excel file dengan lead time summary"""
        try:
            # Prepare data untuk export
            export_data = []
            
            for result in self.results:
                if result['status'] == 'success':
                    row = {
                        'ticket_id': result['ticket_id'],
                        'issue_type': result['final_issue_type'],
                        'main_question': result['main_question'],
                        'performance_rating': result['performance_rating'],
                        'quality_rating': result['quality_rating'],
                        'quality_score': result['quality_score'],
                        'first_reply_lead_time_minutes': result.get('first_reply_lead_time_minutes'),
                        'final_reply_lead_time_minutes': result.get('final_reply_lead_time_minutes'),
                        'first_reply_found': result['first_reply_found'],
                        'final_reply_found': result['final_reply_found'],
                        'customer_leave': result.get('customer_leave', False),
                        'follow_up_ticket': result.get('follow_up_ticket', ''),
                        'recommendation': result['recommendation'],
                        'detection_confidence': result['detection_confidence'],
                        'ml_confidence': result['ml_confidence'],
                        'total_messages': result['total_messages'],
                        'total_qa_pairs': result['total_qa_pairs'],
                        'answered_pairs': result['answered_pairs'],
                        'first_reply_message': result.get('first_reply_message', '')[:100] + '...' if result.get('first_reply_message') else '',
                        'final_reply_message': result.get('final_reply_message', '')[:100] + '...' if result.get('final_reply_message') else ''
                    }
                    export_data.append(row)
                else:
                    row = {
                        'ticket_id': result['ticket_id'],
                        'issue_type': 'FAILED',
                        'main_question': result.get('failure_reason', 'Analysis failed'),
                        'performance_rating': 'N/A',
                        'quality_rating': 'N/A',
                        'quality_score': 0,
                        'first_reply_lead_time_minutes': None,
                        'final_reply_lead_time_minutes': None,
                        'first_reply_found': False,
                        'final_reply_found': False,
                        'customer_leave': False,
                        'follow_up_ticket': '',
                        'recommendation': 'Analysis failed',
                        'detection_confidence': 0,
                        'ml_confidence': 0,
                        'total_messages': 0,
                        'total_qa_pairs': 0,
                        'answered_pairs': 0,
                        'first_reply_message': '',
                        'final_reply_message': ''
                    }
                    export_data.append(row)
            
            # Create DataFrame dan save
            df_export = pd.DataFrame(export_data)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Detailed results
                df_export.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Summary statistics
                summary_data = self._create_summary_sheet()
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Performance metrics
                perf_data = self._create_performance_sheet()
                perf_df = pd.DataFrame(perf_data)
                perf_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # Lead Time Summary - SHEET BARU
                lead_time_data = self._create_lead_time_summary_sheet()
                lead_time_df = pd.DataFrame(lead_time_data)
                lead_time_df.to_excel(writer, sheet_name='Lead_Time_Summary', index=False)
            
            print(f"💾 Results exported to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return None
    
    def _create_lead_time_summary_sheet(self):
        """Create lead time summary sheet - REQUIREMENT BARU"""
        stats = self.analysis_stats
        
        lead_time_data = [
            ['LEAD TIME SUMMARY REPORT', '', '', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '', ''],
            ['', '', '', ''],
            ['OVERALL LEAD TIME STATISTICS', 'First Reply', 'Final Reply', ''],
            ['Average Lead Time (minutes)', 
             f"{stats.get('overall_lead_times', {}).get('first_reply_avg_minutes', 0):.1f}",
             f"{stats.get('overall_lead_times', {}).get('final_reply_avg_minutes', 0):.1f}", ''],
            ['Median Lead Time (minutes)',
             f"{stats.get('overall_lead_times', {}).get('first_reply_median_minutes', 0):.1f}", 
             f"{stats.get('overall_lead_times', {}).get('final_reply_median_minutes', 0):.1f}", ''],
            ['Number of Samples',
             stats.get('overall_lead_times', {}).get('first_reply_samples', 0),
             stats.get('overall_lead_times', {}).get('final_reply_samples', 0), ''],
            ['', '', '', ''],
            ['LEAD TIME BY ISSUE TYPE', 'First Reply Avg (min)', 'Final Reply Avg (min)', 'Samples']
        ]
        
        if 'issue_type_lead_times' in stats:
            for issue_type, lt_stats in stats['issue_type_lead_times'].items():
                first_avg = lt_stats.get('first_reply_avg_minutes', 0)
                final_avg = lt_stats.get('final_reply_avg_minutes', 0)
                
                first_str = f"{first_avg:.1f}" if first_avg is not None else "N/A"
                final_str = f"{final_avg:.1f}" if final_avg is not None else "N/A"
                samples_str = f"F:{lt_stats.get('first_reply_samples', 0)} / R:{lt_stats.get('final_reply_samples', 0)}"
                
                lead_time_data.append([
                    issue_type.upper(),
                    first_str,
                    final_str,
                    samples_str
                ])
        
        return lead_time_data

    def _create_summary_sheet(self):
        """Create summary sheet data"""
        stats = self.analysis_stats
        
        summary_data = [
            ['COMPLETE ANALYSIS PIPELINE - SUMMARY REPORT', '', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''],
            ['', '', ''],
            ['OVERALL STATISTICS', '', ''],
            ['Total Tickets Processed', stats.get('total_tickets', 0), ''],
            ['Successful Analysis', stats.get('successful_analysis', 0), ''],
            ['Failed Analysis', stats.get('failed_analysis', 0), ''],
            ['Success Rate', f"{stats.get('success_rate', 0)*100:.1f}%", ''],
            ['Total Analysis Time', f"{stats.get('analysis_duration_seconds', 0):.1f} seconds", ''],
            ['Average Time per Ticket', f"{stats.get('avg_analysis_time_per_ticket', 0):.2f} seconds", ''],
            ['', '', ''],
            ['LEAD TIME SUMMARY', 'First Reply', 'Final Reply']
        ]
        
        # Add lead time summary
        if 'overall_lead_times' in stats:
            overall_lt = stats['overall_lead_times']
            summary_data.extend([
                ['Average Lead Time (minutes)', 
                 f"{overall_lt.get('first_reply_avg_minutes', 0):.1f}", 
                 f"{overall_lt.get('final_reply_avg_minutes', 0):.1f}"],
                ['Number of Samples', 
                 overall_lt.get('first_reply_samples', 0), 
                 overall_lt.get('final_reply_samples', 0)]
            ])
        
        summary_data.extend([
            ['', '', ''],
            ['ISSUE TYPE DISTRIBUTION', 'Count', 'Percentage']
        ])
        
        if 'issue_type_distribution' in stats:
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats.get('successful_analysis', 1)) * 100
                summary_data.append([issue_type.title(), count, f"{percentage:.1f}%"])
        
        summary_data.extend([
            ['', '', ''],
            ['PERFORMANCE DISTRIBUTION', 'Count', 'Percentage']
        ])
        
        if 'performance_distribution' in stats:
            for rating, count in stats['performance_distribution'].items():
                percentage = (count / stats.get('successful_analysis', 1)) * 100
                summary_data.append([rating.upper(), count, f"{percentage:.1f}%"])
        
        return summary_data
    
    def _create_performance_sheet(self):
        """Create performance metrics sheet"""
        stats = self.analysis_stats
        
        perf_data = [
            ['PERFORMANCE METRICS', 'Value', ''],
            ['Reply Effectiveness', '', ''],
            ['First Reply Found Rate', f"{stats.get('reply_effectiveness', {}).get('first_reply_found_rate', 0)*100:.1f}%", ''],
            ['Final Reply Found Rate', f"{stats.get('reply_effectiveness', {}).get('final_reply_found_rate', 0)*100:.1f}%", ''],
            ['Both Replies Found Rate', f"{stats.get('reply_effectiveness', {}).get('both_replies_found_rate', 0)*100:.1f}%", ''],
            ['Customer Leave Cases', stats.get('reply_effectiveness', {}).get('customer_leave_cases', 0), ''],
            ['Follow-up Cases', stats.get('reply_effectiveness', {}).get('follow_up_cases', 0), ''],
            ['', '', ''],
            ['Lead Time Statistics', '', '']
        ]
        
        if 'overall_lead_times' in stats:
            lt = stats['overall_lead_times']
            perf_data.extend([
                ['Average First Reply Time', f"{lt.get('first_reply_avg_minutes', 0):.2f} minutes", ''],
                ['Average Final Reply Time', f"{lt.get('final_reply_avg_minutes', 0):.2f} minutes", ''],
                ['First Reply Samples', lt.get('first_reply_samples', 0), ''],
                ['Final Reply Samples', lt.get('final_reply_samples', 0), '']
            ])
        
        perf_data.extend([
            ['', '', ''],
            ['Quality Metrics', '', ''],
            ['Average Quality Score', f"{stats.get('quality_stats', {}).get('avg_quality_score', 0):.2f}/6", ''],
            ['Most Common Quality Rating', stats.get('quality_stats', {}).get('avg_quality_rating', 'N/A').upper(), '']
        ])
        
        return perf_data

# Initialize Fixed Pipeline
pipeline = CompleteAnalysisPipeline()

print("✅ FULLY FIXED Complete Analysis Pipeline Ready!")
print("   ✓ All raw data preserved for export")
print("   ✓ Enhanced statistics tracking")
print("   ✓ Customer leave detection")
print("   ✓ Follow-up ticket analysis")
print("   ✓ Lead time summary per issue type")
print("=" * 60)

# Model Training & Evaluation dengan Real Data (FIXED)
class ModelTrainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.training_data = []
        self.evaluation_results = {}
        
    def collect_training_data_from_analysis(self, results):
        """Collect training data dari analysis results"""
        print("📚 Collecting training data from analysis results...")
        
        training_samples = []
        
        for result in results:
            if result['status'] == 'success':
                # Use the final classification sebagai ground truth
                training_samples.append({
                    'text': result['main_question'],
                    'label': result['final_issue_type'],
                    'ticket_id': result['ticket_id'],
                    'confidence': result['detection_confidence'],
                    'ml_confidence': result['ml_confidence']
                })
        
        self.training_data = training_samples
        print(f"✅ Collected {len(training_samples)} training samples")
        return training_samples
    
    def enhance_training_data(self):
        """Enhance training data dengan manual corrections dan additional samples"""
        print("🔄 Enhancing training data...")
        
        # Additional curated samples berdasarkan domain knowledge
        enhanced_samples = [
            # Additional NORMAL samples
            {"text": "berapa harga mobil avanza terbaru", "label": "normal"},
            {"text": "info promo cashback toyota", "label": "normal"},
            {"text": "cara booking test drive fortuner", "label": "normal"},
            {"text": "alamat dealer toyota terdekat", "label": "normal"},
            {"text": "jam operasional bengkel", "label": "normal"},
            {"text": "spesifikasi lengkap rush", "label": "normal"},
            {"text": "syarat kredit mobil baru", "label": "normal"},
            {"text": "beda innova zenix dan innova lama", "label": "normal"},
            
            # Additional SERIOUS samples  
            {"text": "mobil tiba-tiba mati di jalan tol", "label": "serious"},
            {"text": "mesin overheating terus menerus", "label": "serious"},
            {"text": "rem tidak berfungsi dengan baik", "label": "serious"},
            {"text": "aki soak tidak bisa starter", "label": "serious"},
            {"text": "lampu dashboard berkedip semua", "label": "serious"},
            {"text": "oli mesin bocor parah", "label": "serious"},
            {"text": "ban pecah di jalan cepat", "label": "serious"},
            {"text": "mobil terbakar sendiri", "label": "serious"},
            
            # Additional COMPLAINT samples
            {"text": "sangat kecewa dengan pelayanan bengkel", "label": "complaint"},
            {"text": "komplain sparepart palsu yang dipasang", "label": "complaint"},
            {"text": "protes untuk biaya servis yang mahal", "label": "complaint"},
            {"text": "minta refund untuk produk cacat", "label": "complaint"},
            {"text": "kecewa dengan waiting time yang lama", "label": "complaint"},
            {"text": "komplain untuk janji tidak ditepati", "label": "complaint"},
            {"text": "pelayanan customer service sangat buruk", "label": "complaint"},
            {"text": "protes untuk penanganan yang lamban", "label": "complaint"}
        ]
        
        # Combine dengan collected data
        if self.training_data:
            current_texts = [sample['text'] for sample in self.training_data]
            for sample in enhanced_samples:
                if sample['text'] not in current_texts:
                    self.training_data.append(sample)
        else:
            self.training_data = enhanced_samples
        
        print(f"✅ Enhanced training data: {len(self.training_data)} total samples")
        
        # Show distribution
        labels = [sample['label'] for sample in self.training_data]
        distribution = Counter(labels)
        print(f"📊 Training data distribution: {dict(distribution)}")
    
    def train_and_evaluate_model(self, test_size=0.2):
        """Train dan evaluate model dengan real data"""
        if not self.training_data:
            print("❌ No training data available")
            return None
        
        print("🤖 Training model with enhanced real data...")
        
        # Prepare data
        texts = [sample['text'] for sample in self.training_data]
        labels = [sample['label'] for sample in self.training_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=config.RANDOM_STATE,
            stratify=labels
        )
        
        print(f"📊 Dataset: {len(texts)} samples")
        print(f"📊 Training: {len(X_train)} samples")
        print(f"📊 Testing: {len(X_test)} samples")
        print(f"📊 Class distribution: {Counter(labels)}")
        
        # Train model
        success = self.pipeline.classifier.train(X_train, y_train)
        
        if success:
            # Evaluate model
            accuracy = self.pipeline.classifier.evaluate_model(X_test, y_test)
            
            # Store evaluation results
            self.evaluation_results = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': dict(Counter(labels)),
                'test_distribution': dict(Counter(y_test))
            }
            
            # Cross-validation untuk lebih robust evaluation
            cv_results = self._cross_validate(texts, labels)
            self.evaluation_results.update(cv_results)
            
            self._print_evaluation_report()
            
            return accuracy
        else:
            print("❌ Model training failed")
            return None
    
    def _cross_validate(self, texts, labels, cv_folds=5):
        """Perform cross-validation untuk robust evaluation"""
        print("🔍 Performing cross-validation...")
        
        try:
            from sklearn.model_selection import cross_val_score
            
            # Create pipeline untuk cross-validation
            from sklearn.pipeline import Pipeline
            cv_pipeline = Pipeline([
                ('tfidf', self.pipeline.classifier.vectorizer),
                ('clf', self.pipeline.classifier.classifier)
            ])
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                cv_pipeline, texts, labels, 
                cv=cv_folds, scoring='accuracy'
            )
            
            return {
                'cross_val_accuracy_mean': np.mean(cv_scores),
                'cross_val_accuracy_std': np.std(cv_scores),
                'cross_val_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            print(f"⚠️ Cross-validation failed: {e}")
            return {}
    
    def _print_evaluation_report(self):
        """Print comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        eval_results = self.evaluation_results
        
        print(f"🎯 ACCURACY METRICS:")
        print(f"   • Test Accuracy: {eval_results.get('accuracy', 0):.3f}")
        if 'cross_val_accuracy_mean' in eval_results:
            print(f"   • Cross-Val Accuracy: {eval_results['cross_val_accuracy_mean']:.3f} (±{eval_results['cross_val_accuracy_std']:.3f})")
        
        print(f"\n📊 DATASET INFO:")
        print(f"   • Total Samples: {eval_results.get('training_samples', 0) + eval_results.get('test_samples', 0)}")
        print(f"   • Training Samples: {eval_results.get('training_samples', 0)}")
        print(f"   • Test Samples: {eval_results.get('test_samples', 0)}")
        
        if 'class_distribution' in eval_results:
            print(f"   • Class Distribution: {eval_results['class_distribution']}")
        
        if 'cross_val_scores' in eval_results:
            print(f"\n🔍 CROSS-VALIDATION DETAILS:")
            for i, score in enumerate(eval_results['cross_val_scores']):
                print(f"   • Fold {i+1}: {score:.3f}")
        
        # Model performance assessment
        accuracy = eval_results.get('accuracy', 0)
        if accuracy >= 0.9:
            assessment = "EXCELLENT 🎯"
        elif accuracy >= 0.8:
            assessment = "VERY GOOD ✅"  
        elif accuracy >= 0.7:
            assessment = "GOOD 👍"
        elif accuracy >= 0.6:
            assessment = "FAIR ⚠️"
        else:
            assessment = "POOR ❌"
        
        print(f"\n📈 PERFORMANCE ASSESSMENT: {assessment}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if accuracy >= 0.85:
            print(f"   • Model ready for production use")
            print(f"   • Continue monitoring performance")
        elif accuracy >= 0.7:
            print(f"   • Model acceptable for use")
            print(f"   • Consider adding more training data")
        else:
            print(f"   • Model needs improvement")
            print(f"   • Collect more labeled data")
            print(f"   • Review feature engineering")
    
    def analyze_model_confidence(self, results):
        """Analyze model confidence pada real predictions"""
        print("\n🔍 Analyzing model confidence on real data...")
        
        confident_predictions = 0
        total_predictions = 0
        confidence_scores = []
        
        for result in results:
            if result['status'] == 'success':
                total_predictions += 1
                ml_confidence = result.get('ml_confidence', 0)
                confidence_scores.append(ml_confidence)
                
                if ml_confidence > 0.7:  # High confidence threshold
                    confident_predictions += 1
        
        if total_predictions > 0:
            avg_confidence = np.mean(confidence_scores)
            high_confidence_rate = confident_predictions / total_predictions
            
            print(f"   • Total Predictions: {total_predictions}")
            print(f"   • High Confidence Predictions: {confident_predictions} ({high_confidence_rate*100:.1f}%)")
            print(f"   • Average Confidence: {avg_confidence:.3f}")
            print(f"   • Confidence Range: {np.min(confidence_scores):.3f} - {np.max(confidence_scores):.3f}")
            
            return {
                'total_predictions': total_predictions,
                'high_confidence_predictions': confident_predictions,
                'high_confidence_rate': high_confidence_rate,
                'avg_confidence': avg_confidence,
                'confidence_scores': confidence_scores
            }
        
        return {}

    def save_model_report(self):
        """Save model evaluation report ke file"""
        try:
            report_path = "output/model_evaluation_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("🤖 CHAT ANALYSIS MODEL EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Accuracy metrics
                f.write("ACCURACY METRICS:\n")
                f.write(f"- Test Accuracy: {self.evaluation_results.get('accuracy', 0):.3f}\n")
                if 'cross_val_accuracy_mean' in self.evaluation_results:
                    f.write(f"- Cross-Val Accuracy: {self.evaluation_results['cross_val_accuracy_mean']:.3f} (±{self.evaluation_results['cross_val_accuracy_std']:.3f})\n")
                
                # Dataset info
                f.write(f"\nDATASET INFO:\n")
                f.write(f"- Total Samples: {len(self.training_data)}\n")
                f.write(f"- Training Samples: {self.evaluation_results.get('training_samples', 0)}\n")
                f.write(f"- Test Samples: {self.evaluation_results.get('test_samples', 0)}\n")
                f.write(f"- Class Distribution: {self.evaluation_results.get('class_distribution', {})}\n")
                
                # Model info
                f.write(f"\nMODEL INFO:\n")
                f.write(f"- Classifier: Logistic Regression\n")
                f.write(f"- Vectorizer: TF-IDF\n")
                f.write(f"- Features: {self.pipeline.classifier.vectorizer.max_features}\n")
                
                # Recommendations
                accuracy = self.evaluation_results.get('accuracy', 0)
                f.write(f"\nRECOMMENDATIONS:\n")
                if accuracy >= 0.85:
                    f.write("- ✅ Model ready for production use\n")
                elif accuracy >= 0.7:
                    f.write("- ⚠️ Model acceptable, consider more training data\n")
                else:
                    f.write("- ❌ Model needs improvement\n")
            
            print(f"💾 Model evaluation report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ Error saving model report: {e}")

# Initialize Model Trainer
model_trainer = ModelTrainer(pipeline)

print("✅ Model Trainer Ready!")
print("=" * 60)

# Results Export & Visualization
class ResultsExporter:
    def __init__(self):
        self.output_dir = "output/"
        self.reports_dir = f"{self.output_dir}reports/"
        self.visualizations_dir = f"{self.output_dir}visualizations/"
        
        # Colors untuk visualizations
        self.colors = {
            'normal': '#2E86AB',
            'serious': '#A23B72', 
            'complaint': '#F18F01',
            'failed': '#C73E1D'
        }
        
        self.performance_colors = {
            'excellent': '#2E86AB',
            'good': '#A23B72',
            'fair': '#F18F01',
            'poor': '#C73E1D'
        }
        
        # Create directories jika belum ada
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.reports_dir).mkdir(exist_ok=True)
        Path(self.visualizations_dir).mkdir(exist_ok=True)
        
        print("✅ Enhanced Results Exporter Initialized")
    
    def export_comprehensive_results(self, results, stats, filename="comprehensive_analysis_results.xlsx"):
        """Export COMPLETE results dengan semua data parse - FIXED VERSION"""
        output_path = f"{self.output_dir}{filename}"
        
        print(f"💾 Exporting COMPREHENSIVE results to {output_path}...")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Detailed Analysis Results (LENGKAP)
                self._create_comprehensive_detailed_sheet(writer, results)
                
                # Sheet 2: Q-A Pairs Raw Data
                self._create_qa_pairs_sheet(writer, results)
                
                # Sheet 3: Main Issue Analysis Details  
                self._create_main_issue_sheet(writer, results)
                
                # Sheet 4: Reply Analysis Details
                self._create_reply_analysis_sheet(writer, results)
                
                # Sheet 5: Summary Statistics
                self._create_summary_sheet(writer, stats)
                
                # Sheet 6: Performance Metrics
                self._create_performance_sheet(writer, results)
                
                # Sheet 7: Lead Time Analysis
                self._create_lead_time_sheet(writer, results)
                
                # Sheet 8: Quality Assessment
                self._create_quality_sheet(writer, results)
                
                # Sheet 9: Lead Time Summary - BARU
                self._create_lead_time_summary_sheet(writer, stats)
            
            print(f"✅ COMPREHENSIVE results exported: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error exporting comprehensive results: {e}")
            # Fallback: create simple export
            return self._create_fallback_export(results, stats)
        
    def _create_lead_time_summary_sheet(self, writer, stats):
        """Create lead time summary sheet - REQUIREMENT BARU"""
        lead_time_data = [
            ['LEAD TIME SUMMARY REPORT', '', '', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '', ''],
            ['', '', '', ''],
            ['OVERALL LEAD TIME STATISTICS', 'First Reply', 'Final Reply', ''],
            ['Average Lead Time (minutes)', 
             f"{stats.get('overall_lead_times', {}).get('first_reply_avg_minutes', 0):.1f}",
             f"{stats.get('overall_lead_times', {}).get('final_reply_avg_minutes', 0):.1f}", ''],
            ['Median Lead Time (minutes)',
             f"{stats.get('overall_lead_times', {}).get('first_reply_median_minutes', 0):.1f}", 
             f"{stats.get('overall_lead_times', {}).get('final_reply_median_minutes', 0):.1f}", ''],
            ['Number of Samples',
             stats.get('overall_lead_times', {}).get('first_reply_samples', 0),
             stats.get('overall_lead_times', {}).get('final_reply_samples', 0), ''],
            ['', '', '', ''],
            ['LEAD TIME BY ISSUE TYPE', 'First Reply Avg (min)', 'Final Reply Avg (min)', 'Samples']
        ]
        
        if 'issue_type_lead_times' in stats:
            for issue_type, lt_stats in stats['issue_type_lead_times'].items():
                first_avg = lt_stats.get('first_reply_avg_minutes', 'N/A')
                final_avg = lt_stats.get('final_reply_avg_minutes', 'N/A')
                
                first_str = f"{first_avg:.1f}" if first_avg != 'N/A' else 'N/A'
                final_str = f"{final_avg:.1f}" if final_avg != 'N/A' else 'N/A'
                samples_str = f"F:{lt_stats['first_reply_samples']} / R:{lt_stats['final_reply_samples']}"
                
                lead_time_data.append([
                    issue_type.upper(),
                    first_str,
                    final_str,
                    samples_str
                ])
        
        lead_time_df = pd.DataFrame(lead_time_data)
        lead_time_df.to_excel(writer, sheet_name='Lead_Time_Summary', index=False, header=False)

    def _create_qa_pairs_sheet(self, writer, results):
        """Create sheet dengan RAW Q-A PAIRS data - FIXED SORTING VERSION"""
        qa_pairs_data = []
        
        for result in results:
            if result['status'] == 'success' and '_raw_qa_pairs' in result:
                # 🔥 FIX: URUTKAN Q-A PAIRS BERDASARKAN POSITION/WAKTU
                sorted_qa_pairs = sorted(
                    result['_raw_qa_pairs'], 
                    key=lambda x: x.get('position', 0)
                )
                
                for i, qa_pair in enumerate(sorted_qa_pairs):
                    qa_pairs_data.append({
                        'Ticket_ID': result['ticket_id'],
                        'QA_Pair_Index': i + 1,
                        'Question': qa_pair.get('question', ''),
                        'Question_Time': qa_pair.get('question_time'),
                        'Bubble_Count': qa_pair.get('bubble_count', 1),
                        'Is_Answered': qa_pair.get('is_answered', False),
                        'Answer': qa_pair.get('answer', ''),
                        'Answer_Time': qa_pair.get('answer_time'),
                        'Answer_Role': qa_pair.get('answer_role', ''),
                        'Lead_Time_Seconds': qa_pair.get('lead_time_seconds'),
                        'Lead_Time_Minutes': qa_pair.get('lead_time_minutes'),
                        'Lead_Time_HHMMSS': qa_pair.get('lead_time_hhmmss'),
                        'Position_Index': qa_pair.get('position', i)  # Untuk debugging
                    })
        
        if qa_pairs_data:
            # 🔥 FIX: URUTKAN DATA UNTUK EXCEL BERDASARKAN TICKET DAN WAKTU
            df_qa = pd.DataFrame(qa_pairs_data)
            
            # Urutkan berdasarkan Ticket_ID dan Question_Time
            df_qa = df_qa.sort_values(['Ticket_ID', 'Question_Time']).reset_index(drop=True)
            
            # Update QA_Pair_Index yang benar setelah sorting
            df_qa['QA_Pair_Index'] = df_qa.groupby('Ticket_ID').cumcount() + 1
            
            df_qa.to_excel(writer, sheet_name='Raw_QA_Pairs', index=False)
            
            print(f"   ✅ Exported {len(df_qa)} Q-A pairs (sorted by time)")
        else:
            # Create empty sheet jika tidak ada data
            empty_df = pd.DataFrame(['No Q-A pairs data available'])
            empty_df.to_excel(writer, sheet_name='Raw_QA_Pairs', index=False, header=False)

    def _create_comprehensive_detailed_sheet(self, writer, results):
        """Create DETAILED sheet dengan SEMUA data"""
        detailed_data = []
        
        for result in results:
            if result['status'] == 'success':
                row = {
                    # BASIC INFO
                    'Ticket_ID': result['ticket_id'],
                    'Status': 'SUCCESS',
                    'Analysis_Timestamp': result['analysis_timestamp'],
                    
                    # CONVERSATION INFO
                    'Total_Messages': result['total_messages'],
                    'Total_QA_Pairs': result['total_qa_pairs'],
                    'Answered_Pairs': result['answered_pairs'],
                    'Conversation_Duration_Min': result.get('conversation_duration_minutes', 'N/A'),
                    'Customer_Leave': result.get('customer_leave', False),
                    'Follow_Up_Ticket': result.get('follow_up_ticket', ''),
                    
                    # MAIN ISSUE - LENGKAP
                    'Main_Question': result['main_question'],
                    'Main_Question_Time': result.get('main_question_time'),
                    'Detected_Issue_Type': result.get('detected_issue_type', 'N/A'),
                    'Final_Issue_Type': result['final_issue_type'],
                    'Detection_Confidence': result['detection_confidence'],
                    'ML_Prediction': result.get('ml_prediction', 'N/A'),
                    'ML_Confidence': result.get('ml_confidence', 'N/A'),
                    'Main_Issue_Reason': result.get('main_issue_reason', 'N/A'),
                    
                    # FIRST REPLY - LENGKAP
                    'First_Reply_Found': result['first_reply_found'],
                    'First_Reply_Message': result.get('first_reply_message', ''),
                    'First_Reply_Time': result.get('first_reply_time'),
                    'First_Reply_Lead_Time_Min': result.get('first_reply_lead_time_minutes'),
                    'First_Reply_Lead_Time_HHMMSS': result.get('first_reply_lead_time_hhmmss'),
                    
                    # FINAL REPLY - LENGKAP
                    'Final_Reply_Found': result['final_reply_found'],
                    'Final_Reply_Message': result.get('final_reply_message', ''),
                    'Final_Reply_Time': result.get('final_reply_time'),
                    'Final_Reply_Lead_Time_Min': result.get('final_reply_lead_time_minutes'),
                    'Final_Reply_Lead_Time_HHMMSS': result.get('final_reply_lead_time_hhmmss'),
                    'Customer_Leave_Note': result.get('customer_leave_note', ''),
                    
                    # PERFORMANCE METRICS
                    'Performance_Rating': result['performance_rating'],
                    'Response_Efficiency': result.get('response_efficiency', 'N/A'),
                    'Resolution_Efficiency': result.get('resolution_efficiency', 'N/A'),
                    'Quality_Rating': result['quality_rating'],
                    'Quality_Score': result['quality_score'],
                    
                    # THRESHOLD & RECOMMENDATIONS
                    'Threshold_Violations': ', '.join(result['threshold_violations']) if result['threshold_violations'] else 'None',
                    'Recommendation': result['recommendation'],
                    'Missing_Elements': ', '.join(result['missing_elements']) if result['missing_elements'] else 'None'
                }
            else:
                row = {
                    'Ticket_ID': result['ticket_id'],
                    'Status': 'FAILED',
                    'Failure_Reason': result['failure_reason'],
                    'Analysis_Timestamp': result['analysis_timestamp']
                }
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Detailed_Analysis']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    def _create_main_issue_sheet(self, writer, results):
        """Create sheet dengan MAIN ISSUE analysis details"""
        main_issue_data = []
        
        for result in results:
            if result['status'] == 'success' and '_raw_main_issue' in result:
                main_issue = result['_raw_main_issue']
                scoring_details = main_issue.get('scoring_details', {})
                
                main_issue_data.append({
                    'Ticket_ID': result['ticket_id'],
                    'Selected_Question': main_issue.get('question', ''),
                    'Question_Time': main_issue.get('question_time'),
                    'Issue_Type': main_issue.get('issue_type', ''),
                    'Confidence_Score': main_issue.get('confidence_score', 0),
                    'Selection_Reason': main_issue.get('selected_reason', ''),
                    
                    # SCORING DETAILS
                    'Complaint_Keyword_Matches': scoring_details.get('complaint_matches', 0),
                    'Serious_Keyword_Matches': scoring_details.get('serious_matches', 0),
                    'Normal_Keyword_Matches': scoring_details.get('normal_matches', 0),
                    'Is_Initial_Question': scoring_details.get('is_initial_question', False),
                    'Is_Follow_Up': scoring_details.get('is_follow_up', False),
                    
                    # CANDIDATE COUNT
                    'Total_Candidates': len(main_issue.get('all_candidates', [])),
                    'Winning_Score': max([c.get('score', 0) for c in main_issue.get('all_candidates', [])]) if main_issue.get('all_candidates') else 0
                })
        
        if main_issue_data:
            df_main_issue = pd.DataFrame(main_issue_data)
            df_main_issue.to_excel(writer, sheet_name='Main_Issue_Details', index=False)
        else:
            empty_df = pd.DataFrame(['No main issue details available'])
            empty_df.to_excel(writer, sheet_name='Main_Issue_Details', index=False, header=False)

    def _create_reply_analysis_sheet(self, writer, results):
        """Create sheet dengan REPLY ANALYSIS details"""
        reply_analysis_data = []
        
        for result in results:
            if result['status'] == 'success' and '_raw_reply_analysis' in result:
                reply_analysis = result['_raw_reply_analysis']
                lead_times = reply_analysis.get('lead_times', {})
                threshold_checks = reply_analysis.get('threshold_checks', {})
                quality_assessment = reply_analysis.get('quality_assessment', {})
                reply_validation = reply_analysis.get('reply_validation', {})
                performance_analysis = reply_analysis.get('performance_analysis', {})
                
                reply_analysis_data.append({
                    'Ticket_ID': result['ticket_id'],
                    'Issue_Type': reply_analysis.get('issue_type', ''),
                    
                    # LEAD TIMES DETAILS
                    'First_Reply_Lead_Time_Seconds': lead_times.get('first_reply_lead_time_seconds'),
                    'Final_Reply_Lead_Time_Seconds': lead_times.get('final_reply_lead_time_seconds'),
                    'Conversation_Duration_Seconds': lead_times.get('conversation_duration_seconds'),
                    
                    # THRESHOLD CHECKS
                    'Threshold_Violations_Count': len([v for v in threshold_checks.values() if v is True]),
                    'Specific_Threshold_Violations': ', '.join([k for k, v in threshold_checks.items() if v is True]),
                    
                    # QUALITY ASSESSMENT
                    'First_Reply_Quality': quality_assessment.get('first_reply_quality', 'unknown'),
                    'Final_Reply_Quality': quality_assessment.get('final_reply_quality', 'unknown'),
                    'Overall_Quality': quality_assessment.get('overall_quality', 'unknown'),
                    
                    # REPLY VALIDATION
                    'First_Reply_Found': reply_validation.get('first_reply_found', False),
                    'Final_Reply_Found': reply_validation.get('final_reply_found', False),
                    'Validation_Quality_Score': reply_validation.get('quality_score', 0),
                    'Validation_Quality_Rating': reply_validation.get('quality_rating', 'poor'),
                    'Missing_Elements': ', '.join(reply_validation.get('missing_elements', [])),
                    'Validation_Recommendation': reply_validation.get('recommendation', ''),
                    
                    # PERFORMANCE ANALYSIS
                    'Performance_Rating': performance_analysis.get('performance_rating', 'unknown'),
                    'Response_Efficiency': performance_analysis.get('response_efficiency', 'unknown'),
                    'Resolution_Efficiency': performance_analysis.get('resolution_efficiency', 'unknown')
                })
        
        if reply_analysis_data:
            df_reply = pd.DataFrame(reply_analysis_data)
            df_reply.to_excel(writer, sheet_name='Reply_Analysis_Details', index=False)
        else:
            empty_df = pd.DataFrame(['No reply analysis details available'])
            empty_df.to_excel(writer, sheet_name='Reply_Analysis_Details', index=False, header=False)

    def _create_summary_sheet(self, writer, stats):
        """Create summary statistics sheet"""
        summary_data = [
            ['COMPREHENSIVE ANALYSIS SUMMARY - ALL DATA EXPORTED', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Sheets in this File', '8 Sheets: Detailed, QA-Pairs, Main-Issue, Reply-Analysis, Summary, Performance, Lead-Time, Quality'],
            ['', ''],
            ['OVERALL STATISTICS', ''],
            ['Total Tickets Processed', stats['total_tickets']],
            ['Successful Analysis', stats['successful_analysis']],
            ['Failed Analysis', stats['failed_analysis']],
            ['Success Rate', f"{stats['success_rate']*100:.1f}%"],
            ['Analysis Duration', f"{stats['analysis_duration_seconds']:.1f} seconds"],
            ['Average Time per Ticket', f"{stats['avg_analysis_time_per_ticket']:.2f} seconds"],
            ['', ''],
            ['DATA COMPLETENESS', ''],
            ['Total Q-A Pairs Extracted', 'See Raw_QA_Pairs sheet'],
            ['Total Main Issues Identified', 'See Main_Issue_Details sheet'], 
            ['Total Reply Analyses', 'See Reply_Analysis_Details sheet'],
            ['', ''],
            ['EXPORT NOTES', ''],
            ['Sheet 1 - Detailed_Analysis', 'Main results dengan semua field'],
            ['Sheet 2 - Raw_QA_Pairs', 'SEMUA Q-A pairs yang berhasil di-parse'],
            ['Sheet 3 - Main_Issue_Details', 'Detail scoring dan selection main issue'],
            ['Sheet 4 - Reply_Analysis_Details', 'Detail analisis reply dan lead times'],
            ['Sheet 5 - Summary_Statistics', 'Statistik aggregate'],
            ['Sheet 6 - Performance_Metrics', 'Performance metrics per ticket'],
            ['Sheet 7 - Lead_Time_Analysis', 'Analisis lead time detail'],
            ['Sheet 8 - Quality_Assessment', 'Assesment kualitas conversation']
        ]
        
        # Tambahkan statistik biasa
        if 'issue_type_distribution' in stats:
            summary_data.extend([['', ''], ['ISSUE TYPE DISTRIBUTION', '']])
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                summary_data.append([f'{issue_type.title()} Issues', f'{count} ({percentage:.1f}%)'])
        
        if 'performance_distribution' in stats:
            summary_data.extend([['', ''], ['PERFORMANCE DISTRIBUTION', '']])
            for rating, count in stats['performance_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                summary_data.append([f'{rating.upper()} Performance', f'{count} ({percentage:.1f}%)'])
        
        if 'lead_time_stats' in stats:
            lt_stats = stats['lead_time_stats']
            summary_data.extend([['', ''], ['LEAD TIME STATISTICS', '']])
            summary_data.extend([
                ['Average Lead Time', f"{lt_stats['avg_lead_time_minutes']:.2f} minutes"],
                ['Median Lead Time', f"{lt_stats['median_lead_time_minutes']:.2f} minutes"],
                ['Minimum Lead Time', f"{lt_stats['min_lead_time_minutes']:.2f} minutes"],
                ['Maximum Lead Time', f"{lt_stats['max_lead_time_minutes']:.2f} minutes"],
                ['Standard Deviation', f"{lt_stats['std_lead_time_minutes']:.2f} minutes"]
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

    def _create_performance_sheet(self, writer, results):
        """Create performance metrics sheet"""
        successful = [r for r in results if r['status'] == 'success']
        
        perf_data = []
        for result in successful:
            perf_data.append({
                'Ticket_ID': result['ticket_id'],
                'Issue_Type': result['final_issue_type'],
                'Performance_Rating': result['performance_rating'],
                'Response_Efficiency': result.get('response_efficiency', 'N/A'),
                'Resolution_Efficiency': result.get('resolution_efficiency', 'N/A'),
                'First_Reply_Lead_Time_Min': result.get('first_reply_lead_time_minutes'),
                'Final_Reply_Lead_Time_Min': result.get('final_reply_lead_time_minutes'),
                'Threshold_Violations_Count': len(result['threshold_violations']),
                'Specific_Violations': ', '.join(result['threshold_violations']) if result['threshold_violations'] else 'None'
            })
        
        df_perf = pd.DataFrame(perf_data)
        df_perf.to_excel(writer, sheet_name='Performance_Metrics', index=False)

    def _create_lead_time_sheet(self, writer, results):
        """Create lead time analysis sheet"""
        successful = [r for r in results if r['status'] == 'success' and r.get('final_reply_lead_time_minutes')]
        
        lead_time_data = []
        for result in successful:
            lead_time_data.append({
                'Ticket_ID': result['ticket_id'],
                'Issue_Type': result['final_issue_type'],
                'Final_Reply_Lead_Time_Min': result['final_reply_lead_time_minutes'],
                'First_Reply_Lead_Time_Min': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Conversation_Duration_Min': result.get('conversation_duration_minutes', 'N/A'),
                'Performance_Rating': result['performance_rating'],
                'Within_Threshold': 'Yes' if not result['threshold_violations'] else 'No'
            })
        
        df_lead = pd.DataFrame(lead_time_data)
        df_lead.to_excel(writer, sheet_name='Lead_Time_Analysis', index=False)

    def _create_quality_sheet(self, writer, results):
        """Create quality assessment sheet"""
        successful = [r for r in results if r['status'] == 'success']
        
        quality_data = []
        for result in successful:
            quality_data.append({
                'Ticket_ID': result['ticket_id'],
                'Issue_Type': result['final_issue_type'],
                'Quality_Rating': result['quality_rating'],
                'Quality_Score': result['quality_score'],
                'First_Reply_Found': 'Yes' if result['first_reply_found'] else 'No',
                'Final_Reply_Found': 'Yes' if result['final_reply_found'] else 'No',
                'Missing_Elements': ', '.join(result['missing_elements']) if result['missing_elements'] else 'None',
                'Recommendation': result['recommendation']
            })
        
        df_quality = pd.DataFrame(quality_data)
        df_quality.to_excel(writer, sheet_name='Quality_Assessment', index=False)

    def create_comprehensive_visualizations(self, results, stats):
        """Create comprehensive visualizations dashboard dengan lead time analysis"""
        print("📊 Creating comprehensive visualizations...")
        
        successful = [r for r in results if r['status'] == 'success']
        
        if not successful:
            print("❌ No successful analyses to visualize")
            return
        
        # Create figure dengan multiple subplots
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle('Chat Analysis Dashboard - Comprehensive Overview', fontsize=16, fontweight='bold')
        
        # Define grid layout
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(4, 3, figure=fig)
        
        # Plot 1: Issue Type Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_issue_type_distribution(ax1, stats)
        
        # Plot 2: Performance Rating Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_performance_distribution(ax2, stats)
        
        # Plot 3: Lead Time Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_lead_time_distribution(ax3, successful)
        
        # Plot 4: Quality Score Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_quality_distribution(ax4, successful)
        
        # Plot 5: Reply Effectiveness
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_reply_effectiveness(ax5, stats)
        
        # Plot 6: Lead Time by Issue Type - PLOT BARU
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_lead_time_comparison(ax6, stats)
        
        # Plot 7: Customer Leave & Follow-up Cases
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_special_cases(ax7, stats)
        
        # Plot 8: Lead Time by Issue Type (Box Plot)
        ax8 = fig.add_subplot(gs[2, 1:])
        self._plot_lead_time_by_issue_type(ax8, successful)
        
        # Plot 9: Performance by Issue Type
        ax9 = fig.add_subplot(gs[3, :])
        self._plot_performance_by_issue_type(ax9, successful)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save dashboard
        dashboard_path = f"{self.visualizations_dir}analysis_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved: {dashboard_path}")
        
        # Create additional individual charts
        self._create_individual_charts(successful, stats)

    def _plot_lead_time_comparison(self, ax, stats):
        """Plot perbandingan lead time first vs final reply"""
        if 'overall_lead_times' not in stats:
            ax.text(0.5, 0.5, 'No lead time data', ha='center', va='center')
            return
        
        overall_lt = stats['overall_lead_times']
        
        categories = ['First Reply', 'Final Reply']
        times = [overall_lt['first_reply_avg_minutes'], overall_lt['final_reply_avg_minutes']]
        samples = [overall_lt['first_reply_samples'], overall_lt['final_reply_samples']]
        
        bars = ax.bar(categories, times, color=['#2E86AB', '#A23B72'])
        ax.set_title('Average Lead Time: First vs Final Reply', fontweight='bold')
        ax.set_ylabel('Average Lead Time (minutes)')
        
        # Add value labels on bars
        for i, (bar, time_val, sample) in enumerate(zip(bars, times, samples)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{time_val:.1f} min\n(n={sample})', 
                   ha='center', va='bottom', fontsize=9)

    def _plot_special_cases(self, ax, stats):
        """Plot special cases: customer leave dan follow-up"""
        if 'reply_effectiveness' not in stats:
            ax.text(0.5, 0.5, 'No special cases data', ha='center', va='center')
            return
        
        eff = stats['reply_effectiveness']
        
        categories = ['Customer Leave', 'Follow-up Cases']
        counts = [eff['customer_leave_cases'], eff['follow_up_cases']]
        
        bars = ax.bar(categories, counts, color=['#F18F01', '#A23B72'])
        ax.set_title('Special Conversation Cases', fontweight='bold')
        ax.set_ylabel('Number of Cases')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')

    def _plot_lead_time_by_issue_type(self, ax, successful):
        """Plot lead time distribution by issue type"""
        issue_data = {}
        
        for result in successful:
            issue_type = result['final_issue_type']
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            if issue_type not in issue_data:
                issue_data[issue_type] = {
                    'first_reply_times': [],
                    'final_reply_times': []
                }
            
            if first_lt is not None:
                issue_data[issue_type]['first_reply_times'].append(first_lt)
            if final_lt is not None:
                issue_data[issue_type]['final_reply_times'].append(final_lt)
        
        if not issue_data:
            ax.text(0.5, 0.5, 'No lead time data by issue type', ha='center', va='center')
            return
        
        # Prepare data untuk grouped bar chart
        labels = list(issue_data.keys())
        first_means = [np.mean(data['first_reply_times']) if data['first_reply_times'] else 0 for data in issue_data.values()]
        final_means = [np.mean(data['final_reply_times']) if data['final_reply_times'] else 0 for data in issue_data.values()]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, first_means, width, label='First Reply', color='#2E86AB')
        bars2 = ax.bar(x + width/2, final_means, width, label='Final Reply', color='#A23B72')
        
        ax.set_title('Average Lead Time by Issue Type', fontweight='bold')
        ax.set_ylabel('Average Lead Time (minutes)')
        ax.set_xticks(x)
        ax.set_xticklabels([label.upper() for label in labels])
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    def _plot_performance_by_issue_type(self, ax, successful):
        """Plot performance rating by issue type"""
        performance_by_type = {}
        for result in successful:
            issue_type = result['final_issue_type']
            performance = result['performance_rating']
            if issue_type not in performance_by_type:
                performance_by_type[issue_type] = []
            performance_by_type[issue_type].append(performance)
        
        # Convert to counts
        plot_data = {}
        for issue_type, performances in performance_by_type.items():
            plot_data[issue_type] = Counter(performances)
        
        # Create stacked bar chart
        ratings = ['excellent', 'good', 'fair', 'poor']
        bottom = np.zeros(len(plot_data))
        
        for i, rating in enumerate(ratings):
            counts = [plot_data[issue_type].get(rating, 0) for issue_type in plot_data.keys()]
            ax.bar(plot_data.keys(), counts, bottom=bottom, label=rating.capitalize(), 
                  color=self.performance_colors.get(rating, '#999999'))
            bottom += counts
        
        ax.set_title('Performance Rating by Issue Type', fontweight='bold')
        ax.set_ylabel('Number of Tickets')
        ax.legend()
        plt.xticks(rotation=45)

    def _create_fallback_export(self, results, stats):
        """Create fallback export jika export utama gagal"""
        try:
            simple_data = []
            for result in results:
                if result['status'] == 'success':
                    simple_data.append({
                        'ticket_id': result['ticket_id'],
                        'issue_type': result['final_issue_type'],
                        'main_question': result['main_question'],
                        'first_reply_found': result['first_reply_found'],
                        'final_reply_found': result['final_reply_found'],
                        'first_reply_lead_time': result.get('first_reply_lead_time_minutes'),
                        'final_reply_lead_time': result.get('final_reply_lead_time_minutes'),
                        'performance': result['performance_rating']
                    })
            
            df_simple = pd.DataFrame(simple_data)
            fallback_path = f"{self.output_dir}fallback_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df_simple.to_excel(fallback_path, index=False)
            print(f"✅ Fallback export created: {fallback_path}")
            return fallback_path
        except Exception as e:
            print(f"❌ Fallback export also failed: {e}")
            return None

    def _plot_issue_type_distribution(self, ax, stats):
        """Plot issue type distribution"""
        if 'issue_type_distribution' not in stats:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        types = list(stats['issue_type_distribution'].keys())
        counts = list(stats['issue_type_distribution'].values())
        colors = [self.colors.get(t, '#999999') for t in types]
        
        ax.pie(counts, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Issue Type Distribution', fontweight='bold')
    
    def _plot_performance_distribution(self, ax, stats):
        """Plot performance rating distribution"""
        if 'performance_distribution' not in stats:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        ratings = list(stats['performance_distribution'].keys())
        counts = list(stats['performance_distribution'].values())
        colors = [self.performance_colors.get(r, '#999999') for r in ratings]
        
        bars = ax.bar(ratings, counts, color=colors)
        ax.set_title('Performance Rating Distribution', fontweight='bold')
        ax.set_ylabel('Number of Tickets')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_lead_time_distribution(self, ax, successful):
        """Plot lead time distribution"""
        lead_times = [r.get('final_reply_lead_time_minutes', 0) for r in successful 
                     if r.get('final_reply_lead_time_minutes') is not None]
        
        if not lead_times:
            ax.text(0.5, 0.5, 'No lead time data', ha='center', va='center')
            return
        
        ax.hist(lead_times, bins=15, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax.set_title('Final Reply Lead Time Distribution', fontweight='bold')
        ax.set_xlabel('Lead Time (minutes)')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        avg_lt = np.mean(lead_times)
        ax.axvline(avg_lt, color='red', linestyle='--', label=f'Average: {avg_lt:.1f} min')
        ax.legend()
    
    def _plot_quality_distribution(self, ax, successful):
        """Plot quality score distribution"""
        quality_scores = [r.get('quality_score', 0) for r in successful]
        
        ax.hist(quality_scores, bins=range(0, 8), alpha=0.7, color='#F18F01', edgecolor='black')
        ax.set_title('Quality Score Distribution', fontweight='bold')
        ax.set_xlabel('Quality Score (0-6)')
        ax.set_ylabel('Number of Tickets')
        ax.set_xticks(range(0, 7))
    
    def _plot_reply_effectiveness(self, ax, stats):
        """Plot reply effectiveness metrics"""
        if 'reply_effectiveness' not in stats:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        eff = stats['reply_effectiveness']
        metrics = ['First Reply\nFound', 'Final Reply\nFound', 'Both Replies\nFound']
        rates = [eff['first_reply_found_rate'] * 100, 
                eff['final_reply_found_rate'] * 100,
                eff['both_replies_found_rate'] * 100]
        
        bars = ax.bar(metrics, rates, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_title('Reply Effectiveness Rates', fontweight='bold')
        ax.set_ylabel('Rate (%)')
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom')
    
    def _plot_threshold_violations(self, ax, stats):
        """Plot threshold violations"""
        if 'threshold_violations' not in stats or not stats['threshold_violations']:
            ax.text(0.5, 0.5, 'No threshold violations', ha='center', va='center', fontweight='bold')
            return
        
        violations = list(stats['threshold_violations'].keys())
        counts = list(stats['threshold_violations'].values())
        
        bars = ax.bar(violations, counts, color='#C73E1D')
        ax.set_title('Threshold Violations', fontweight='bold')
        ax.set_ylabel('Number of Occurrences')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_lead_time_by_issue_type(self, ax, successful):
        """Plot lead time by issue type"""
        issue_data = {}
        for result in successful:
            issue_type = result['final_issue_type']
            lead_time = result.get('final_reply_lead_time_minutes')
            if lead_time is not None:
                if issue_type not in issue_data:
                    issue_data[issue_type] = []
                issue_data[issue_type].append(lead_time)
        
        if not issue_data:
            ax.text(0.5, 0.5, 'No lead time data by issue type', ha='center', va='center')
            return
        
        # Prepare data for box plot
        labels = list(issue_data.keys())
        data = [issue_data[label] for label in labels]
        colors = [self.colors.get(label, '#999999') for label in labels]
        
        box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Lead Time Distribution by Issue Type', fontweight='bold')
        ax.set_ylabel('Lead Time (minutes)')
        ax.grid(True, alpha=0.3)
    
    def _create_individual_charts(self, successful, stats):
        """Create additional individual charts"""
        # 1. Performance by Issue Type
        fig, ax = plt.subplots(figsize=(10, 6))
        
        performance_by_type = {}
        for result in successful:
            issue_type = result['final_issue_type']
            performance = result['performance_rating']
            if issue_type not in performance_by_type:
                performance_by_type[issue_type] = []
            performance_by_type[issue_type].append(performance)
        
        # Convert to counts
        plot_data = {}
        for issue_type, performances in performance_by_type.items():
            plot_data[issue_type] = Counter(performances)
        
        # Create stacked bar chart
        ratings = ['excellent', 'good', 'fair', 'poor']
        bottom = np.zeros(len(plot_data))
        
        for i, rating in enumerate(ratings):
            counts = [plot_data[issue_type].get(rating, 0) for issue_type in plot_data.keys()]
            ax.bar(plot_data.keys(), counts, bottom=bottom, label=rating.capitalize(), 
                  color=self.performance_colors.get(rating, '#999999'))
            bottom += counts
        
        ax.set_title('Performance Rating by Issue Type', fontweight='bold')
        ax.set_ylabel('Number of Tickets')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_dir}performance_by_issue_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Individual charts created in {self.visualizations_dir}")

# Initialize Enhanced Results Exporter
exporter = ResultsExporter()

print("✅ ENHANCED Results Exporter & Visualizer Ready!")
print("   ✓ 9 Sheets Excel Export")
print("   ✓ Complete Q-A Pairs Data") 
print("   ✓ Main Issue Scoring Details")
print("   ✓ Reply Analysis Details")
print("   ✓ Lead Time Summary Sheets")
print("=" * 60)

# DEBUG HELPER FUNCTIONS
def debug_problematic_tickets():
    """Debug khusus untuk ticket yang bermasalah"""
    if raw_df is None:
        return
    
    problematic_tickets = ['18ea89910d0d8eac44aecca81d779e3a', '9842cd7eee5451283f8430fb83469940']
    
    print("🐛 DETAILED DEBUG FOR PROBLEMATIC TICKETS")
    print("=" * 60)
    
    parser = ConversationParser()
    for ticket_id in problematic_tickets:
        print(f"\n🔍 DEBUG TICKET: {ticket_id}")
        ticket_df = raw_df[raw_df['Ticket Number'] == ticket_id].sort_values('parsed_timestamp')
        
        # Show ALL messages setelah conversation start
        conversation_start = parser.detect_conversation_start(ticket_df)
        
        if conversation_start:
            filtered_df = parser._filter_bot_messages(ticket_df)
            filtered_df = filtered_df[filtered_df['parsed_timestamp'] >= conversation_start]
            
            print(f"📋 ALL MESSAGES AFTER CONVERSATION START ({len(filtered_df)} messages):")
            for idx, row in filtered_df.iterrows():
                role = row['Role']
                message = str(row['Message'])[:80]
                timestamp = row['parsed_timestamp']
                is_meaningful = parser._is_meaningful_message(message)
                meaningful_flag = "✅" if is_meaningful else "❌"
                
                print(f"   {meaningful_flag} {timestamp} | {role:15} | {message}...")
            
            # Analyze why no Q-A pairs
            print(f"\n🔎 ANALYSIS:")
            customer_msgs = filtered_df[filtered_df['Role'].str.lower().str.contains('customer', na=False)]
            operator_msgs = filtered_df[filtered_df['Role'].str.lower().str.contains('operator|agent', na=False)]
            
            print(f"   • Customer messages: {len(customer_msgs)}")
            print(f"   • Operator messages: {len(operator_msgs)}")
            
            meaningful_customer = [msg for msg in customer_msgs['Message'] if parser._is_meaningful_message(str(msg))]
            print(f"   • Meaningful customer messages: {len(meaningful_customer)}")
            
            if len(meaningful_customer) == 0:
                print("   ❌ REASON: No meaningful customer questions after operator greeting")
            elif len(operator_msgs) == 0:
                print("   ❌ REASON: No operator responses after customer questions")
            else:
                print("   ❌ REASON: Timing/sequence issues in Q-A matching")

def debug_timestamp_issues(results):
    """Debug function untuk investigasi timestamp issues"""
    print("\n🐛 DEBUG TIMESTAMP ISSUES")
    print("=" * 50)
    
    problematic_tickets = []
    
    for result in results:
        if result['status'] == 'success':
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            # Cari tickets dengan lead time negatif
            if (first_lt is not None and first_lt < 0) or (final_lt is not None and final_lt < 0):
                problematic_tickets.append({
                    'ticket_id': result['ticket_id'],
                    'main_question': result['main_question'][:50] + '...',
                    'main_question_time': result.get('main_question_time'),
                    'first_reply_time': result.get('first_reply_time'),
                    'final_reply_time': result.get('final_reply_time'),
                    'first_lead_time': first_lt,
                    'final_lead_time': final_lt
                })
    
    if problematic_tickets:
        print(f"❌ Found {len(problematic_tickets)} tickets with negative lead times")
        for ticket in problematic_tickets:
            print(f"   🎫 {ticket['ticket_id']}:")
            print(f"      First LT: {ticket['first_lead_time']} min")
            print(f"      Final LT: {ticket['final_lead_time']} min")
    else:
        print("✅ No timestamp issues found")

def debug_follow_up_analysis(results, customer_info):
    """Debug follow-up ticket analysis"""
    print("\n🐛 DEBUG FOLLOW-UP ANALYSIS")
    print("=" * 50)
    
    follow_up_cases = [r for r in results if r.get('follow_up_ticket')]
    customer_leave_cases = [r for r in results if r.get('customer_leave')]
    
    print(f"📊 Follow-up Cases: {len(follow_up_cases)}")
    print(f"📊 Customer Leave Cases: {len(customer_leave_cases)}")
    
    for case in follow_up_cases[:3]:  # Show first 3 cases
        print(f"\n🔍 Follow-up Case:")
        print(f"   Original Ticket: {case['ticket_id']}")
        print(f"   Follow-up Ticket: {case['follow_up_ticket']}")
        print(f"   Issue Type: {case['final_issue_type']}")
        print(f"   Main Question: {case['main_question'][:80]}...")

def debug_ticket_analysis(ticket_id, df):
    """Debug detailed analysis untuk ticket tertentu"""
    print(f"\n🔍 DEBUG TICKET: {ticket_id}")
    print("=" * 60)
    
    ticket_df = df[df['Ticket Number'] == ticket_id].sort_values('parsed_timestamp')
    
    # Show semua messages
    print("📋 ALL MESSAGES:")
    for idx, row in ticket_df.iterrows():
        role = row['Role']
        message = str(row['Message'])[:100]
        timestamp = row['parsed_timestamp']
        print(f"   {idx:2d} | {timestamp} | {role:15} | {message}...")
    
    # Parse conversation
    parser = ConversationParser()
    qa_pairs = parser.parse_conversation(ticket_df)
    
    print(f"\n📊 Q-A PAIRS FOUND: {len(qa_pairs)}")
    for i, pair in enumerate(qa_pairs):
        status = "✅ ANSWERED" if pair['is_answered'] else "❌ UNANSWERED"
        print(f"\n{i+1}. {status}")
        print(f"   Q: {pair['question'][:100]}...")
        if pair['is_answered']:
            print(f"   A: {pair['answer'][:100]}...")
            print(f"   Lead Time: {pair.get('lead_time_minutes', 'N/A')} min")
    
    # Analyze replies
    if qa_pairs:
        issue_detector = MainIssueDetector()
        main_issue = issue_detector.detect_main_issue(qa_pairs)
        
        if main_issue:
            print(f"\n🎯 MAIN ISSUE: {main_issue['issue_type']}")
            print(f"   Question: {main_issue['question'][:100]}...")
            
            reply_analyzer = ReplyAnalyzer()
            first_reply, final_reply, analysis = reply_analyzer.analyze_replies(qa_pairs, main_issue['issue_type'])
            
            print(f"\n🔍 REPLY ANALYSIS:")
            print(f"   First Reply Found: {analysis['reply_validation']['first_reply_found']}")
            print(f"   Final Reply Found: {analysis['reply_validation']['final_reply_found']}")
            print(f"   Requirement Compliant: {analysis['requirement_compliant']}")

# Test dengan sample data
if __name__ == "__main__":
    print("🧪 TESTING ENHANCED PIPELINE...")
    
    # Test preprocessor
    preprocessor = DataPreprocessor()
    raw_df = preprocessor.load_raw_data(config.RAW_DATA_PATH)
    
    if raw_df is not None:
        print(f"📊 Data preview:")
        print(f"   Columns: {list(raw_df.columns)}")
        print(f"   Shape: {raw_df.shape}")
        print(f"   Ticket count: {raw_df['Ticket Number'].nunique()}")
        
        # Test dengan sample tickets
        sample_tickets = raw_df['Ticket Number'].unique()[:3]
        print(f"\n🔍 Testing with {len(sample_tickets)} sample tickets...")
        
        for ticket_id in sample_tickets:
            ticket_df = raw_df[raw_df['Ticket Number'] == ticket_id]
            result = pipeline.analyze_single_ticket(ticket_df, ticket_id)
            
            if result['status'] == 'success':
                print(f"   ✅ {ticket_id}: {result['final_issue_type']} - {result['performance_rating']}")
            else:
                print(f"   ❌ {ticket_id}: {result['failure_reason']}")
    
    print("\n🎯 ENHANCED PIPELINE READY FOR PRODUCTION!")

# DEBUG HELPER FUNCTIONS
def debug_ticket_analysis(ticket_id, df):
    """Debug detailed analysis untuk ticket tertentu"""
    print(f"\n🔍 DEBUG TICKET: {ticket_id}")
    print("=" * 60)
    
    ticket_df = df[df['Ticket Number'] == ticket_id].sort_values('parsed_timestamp')
    
    # Show semua messages
    print("📋 ALL MESSAGES:")
    for idx, row in ticket_df.iterrows():
        role = row['Role']
        message = str(row['Message'])[:100]
        timestamp = row['parsed_timestamp']
        print(f"   {idx:2d} | {timestamp} | {role:15} | {message}...")
    
    # Parse conversation
    parser = ConversationParser()
    qa_pairs = parser.parse_conversation(ticket_df)
    
    print(f"\n📊 Q-A PAIRS FOUND: {len(qa_pairs)}")
    for i, pair in enumerate(qa_pairs):
        status = "✅ ANSWERED" if pair['is_answered'] else "❌ UNANSWERED"
        print(f"\n{i+1}. {status}")
        print(f"   Q: {pair['question'][:100]}...")
        if pair['is_answered']:
            print(f"   A: {pair['answer'][:100]}...")
            print(f"   Lead Time: {pair.get('lead_time_minutes', 'N/A')} min")

# Test dengan sample data
if __name__ == "__main__":
    print("🧪 TESTING ENHANCED PIPELINE...")
    
    # Test preprocessor
    preprocessor = DataPreprocessor()
    raw_df = preprocessor.load_raw_data(config.RAW_DATA_PATH)
    
    if raw_df is not None:
        print(f"📊 Data preview:")
        print(f"   Columns: {list(raw_df.columns)}")
        print(f"   Shape: {raw_df.shape}")
        print(f"   Ticket count: {raw_df['Ticket Number'].nunique()}")
        
        # Test dengan sample tickets
        sample_tickets = raw_df['Ticket Number'].unique()[:3]
        print(f"\n🔍 Testing with {len(sample_tickets)} sample tickets...")
        
        for ticket_id in sample_tickets:
            ticket_df = raw_df[raw_df['Ticket Number'] == ticket_id]
            result = pipeline.analyze_single_ticket(ticket_df, ticket_id)
            
            if result['status'] == 'success':
                print(f"   ✅ {ticket_id}: {result['final_issue_type']} - {result['performance_rating']}")
            else:
                print(f"   ❌ {ticket_id}: {result['failure_reason']}")
    
    print("\n🎯 ENHANCED PIPELINE READY FOR PRODUCTION!")

# DEBUG HELPER FUNCTIONS
def debug_problematic_tickets():
    """Debug khusus untuk ticket yang bermasalah"""
    if raw_df is None:
        return
    
    problematic_tickets = ['18ea89910d0d8eac44aecca81d779e3a', '9842cd7eee5451283f8430fb83469940']
    
    print("🐛 DETAILED DEBUG FOR PROBLEMATIC TICKETS")
    print("=" * 60)
    
    parser = ConversationParser()
    for ticket_id in problematic_tickets:
        print(f"\n🔍 DEBUG TICKET: {ticket_id}")
        ticket_df = raw_df[raw_df['Ticket Number'] == ticket_id].sort_values('parsed_timestamp')
        
        # Show ALL messages setelah conversation start
        conversation_start = parser.detect_conversation_start(ticket_df)
        
        if conversation_start:
            filtered_df = parser._filter_bot_messages(ticket_df)
            filtered_df = filtered_df[filtered_df['parsed_timestamp'] >= conversation_start]
            
            print(f"📋 ALL MESSAGES AFTER CONVERSATION START ({len(filtered_df)} messages):")
            for idx, row in filtered_df.iterrows():
                role = row['Role']
                message = str(row['Message'])[:80]
                timestamp = row['parsed_timestamp']
                is_meaningful = parser._is_meaningful_message(message)
                meaningful_flag = "✅" if is_meaningful else "❌"
                
                print(f"   {meaningful_flag} {timestamp} | {role:15} | {message}...")
            
            # Analyze why no Q-A pairs
            print(f"\n🔎 ANALYSIS:")
            customer_msgs = filtered_df[filtered_df['Role'].str.lower().str.contains('customer', na=False)]
            operator_msgs = filtered_df[filtered_df['Role'].str.lower().str.contains('operator|agent', na=False)]
            
            print(f"   • Customer messages: {len(customer_msgs)}")
            print(f"   • Operator messages: {len(operator_msgs)}")
            
            meaningful_customer = [msg for msg in customer_msgs['Message'] if parser._is_meaningful_message(str(msg))]
            print(f"   • Meaningful customer messages: {len(meaningful_customer)}")
            
            if len(meaningful_customer) == 0:
                print("   ❌ REASON: No meaningful customer questions after operator greeting")
            elif len(operator_msgs) == 0:
                print("   ❌ REASON: No operator responses after customer questions")
            else:
                print("   ❌ REASON: Timing/sequence issues in Q-A matching")

def debug_timestamp_issues(results):
    """Debug function untuk investigasi timestamp issues"""
    print("\n🐛 DEBUG TIMESTAMP ISSUES")
    print("=" * 50)
    
    problematic_tickets = []
    
    for result in results:
        if result['status'] == 'success':
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            # Cari tickets dengan lead time negatif
            if (first_lt is not None and first_lt < 0) or (final_lt is not None and final_lt < 0):
                problematic_tickets.append({
                    'ticket_id': result['ticket_id'],
                    'main_question': result['main_question'][:50] + '...',
                    'main_question_time': result.get('main_question_time'),
                    'first_reply_time': result.get('first_reply_time'),
                    'final_reply_time': result.get('final_reply_time'),
                    'first_lead_time': first_lt,
                    'final_lead_time': final_lt
                })
    
    if problematic_tickets:
        print(f"❌ Found {len(problematic_tickets)} tickets with negative lead times")
        for ticket in problematic_tickets:
            print(f"   🎫 {ticket['ticket_id']}:")
            print(f"      First LT: {ticket['first_lead_time']} min")
            print(f"      Final LT: {ticket['final_lead_time']} min")
    else:
        print("✅ No timestamp issues found")

def debug_follow_up_analysis(results, customer_info):
    """Debug follow-up ticket analysis"""
    print("\n🐛 DEBUG FOLLOW-UP ANALYSIS")
    print("=" * 50)
    
    follow_up_cases = [r for r in results if r.get('follow_up_ticket')]
    customer_leave_cases = [r for r in results if r.get('customer_leave')]
    
    print(f"📊 Follow-up Cases: {len(follow_up_cases)}")
    print(f"📊 Customer Leave Cases: {len(customer_leave_cases)}")
    
    for case in follow_up_cases[:3]:  # Show first 3 cases
        print(f"\n🔍 Follow-up Case:")
        print(f"   Original Ticket: {case['ticket_id']}")
        print(f"   Follow-up Ticket: {case['follow_up_ticket']}")
        print(f"   Issue Type: {case['final_issue_type']}")
        print(f"   Main Question: {case['main_question'][:80]}...")

# Test dengan sample data
if __name__ == "__main__":
    print("🧪 TESTING ENHANCED PIPELINE...")
    
    # Test preprocessor
    preprocessor = DataPreprocessor()
    raw_df = preprocessor.load_raw_data(config.RAW_DATA_PATH)
    
    if raw_df is not None:
        print(f"📊 Data preview:")
        print(f"   Columns: {list(raw_df.columns)}")
        print(f"   Shape: {raw_df.shape}")
        print(f"   Ticket count: {raw_df['Ticket Number'].nunique()}")
        
        # Test dengan sample tickets
        sample_tickets = raw_df['Ticket Number'].unique()[:3]
        print(f"\n🔍 Testing with {len(sample_tickets)} sample tickets...")
        
        for ticket_id in sample_tickets:
            ticket_df = raw_df[raw_df['Ticket Number'] == ticket_id]
            result = pipeline.analyze_single_ticket(ticket_df, ticket_id)
            
            if result['status'] == 'success':
                print(f"   ✅ {ticket_id}: {result['final_issue_type']} - {result['performance_rating']}")
            else:
                print(f"   ❌ {ticket_id}: {result['failure_reason']}")
    
    print("\n🎯 ENHANCED PIPELINE READY FOR PRODUCTION!")

def debug_ticket_analysis(ticket_id, df):
    """Debug detailed analysis untuk ticket tertentu"""
    print(f"\n🔍 DEBUG TICKET: {ticket_id}")
    print("=" * 60)
    
    ticket_df = df[df['Ticket Number'] == ticket_id].sort_values('parsed_timestamp')
    
    # Show semua messages
    print("📋 ALL MESSAGES:")
    for idx, row in ticket_df.iterrows():
        role = row['Role']
        message = str(row['Message'])[:100]
        timestamp = row['parsed_timestamp']
        print(f"   {idx:2d} | {timestamp} | {role:15} | {message}...")
    
    # Parse conversation
    parser = ConversationParser()
    qa_pairs = parser.parse_conversation(ticket_df)
    
    print(f"\n📊 Q-A PAIRS FOUND: {len(qa_pairs)}")
    for i, pair in enumerate(qa_pairs):
        status = "✅ ANSWERED" if pair['is_answered'] else "❌ UNANSWERED"
        print(f"\n{i+1}. {status}")
        print(f"   Q: {pair['question'][:100]}...")
        if pair['is_answered']:
            print(f"   A: {pair['answer'][:100]}...")
            print(f"   Lead Time: {pair.get('lead_time_minutes', 'N/A')} min")
    
    # Analyze replies
    if qa_pairs:
        issue_detector = MainIssueDetector()
        main_issue = issue_detector.detect_main_issue(qa_pairs)
        
        if main_issue:
            print(f"\n🎯 MAIN ISSUE: {main_issue['issue_type']}")
            print(f"   Question: {main_issue['question'][:100]}...")
            
            reply_analyzer = ReplyAnalyzer()
            first_reply, final_reply, analysis = reply_analyzer.analyze_replies(qa_pairs, main_issue['issue_type'])
            
            print(f"\n🔍 REPLY ANALYSIS:")
            print(f"   First Reply Found: {analysis['reply_validation']['first_reply_found']}")
            print(f"   Final Reply Found: {analysis['reply_validation']['final_reply_found']}")
            print(f"   Requirement Compliant: {analysis['requirement_compliant']}")




