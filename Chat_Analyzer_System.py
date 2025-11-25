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
    COMPLAINT_DATA_PATH = "data/complaint_data.xlsx"
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
    ABANDONED_TIMEOUT_MINUTES = 30  # 30 menit tanpa response dari customer
    CUSTOMER_LEAVE_TIMEOUT = 30  # 30 menit untuk detect customer leave
    
    # Keywords
    TICKET_REOPENED_KEYWORD = "Ticket Has Been Reopened by"
    CUSTOMER_LEAVE_KEYWORD = "Mohon maaf, dikarenakan tidak ada respon, chat ini Kami akhiri. Terima kasih telah menggunakan layanan Live Chat Toyota Astra Motor, selamat beraktivitas kembali."
    OPERATOR_GREETING_KEYWORDS = [
        "Selamat pagi", "Selamat siang", "Selamat sore", "Selamat malam",
        "Selamat datang di layanan Live Chat Toyota Astra Motor"
    ]
    
    # Action keywords untuk serious first reply
    ACTION_KEYWORDS = [
        "diteruskan", "disampaikan", "dihubungi", "dicek", "dipelajari",
        "ditindaklanjuti", "dilakukan pengecekan", "dibantu", "dikonsultasikan",
        "dikoordinasikan", "dilaporkan", "dievaluasi", "dianalisis"
    ]

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
            'ticket automation': 'Ticket Automation',
            '': 'Blank'
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
    
    def load_complaint_data(self, file_path):
        """Load complaint data untuk matching"""
        print(f"📖 Loading complaint data from {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✅ Loaded {len(df)} complaint records")
            
            # Clean phone numbers
            if 'No.Handphone' in df.columns:
                df['Cleaned_Phone'] = df['No.Handphone'].astype(str).str.replace(r'\D', '', regex=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading complaint data: {e}")
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
        
        # Fill blank roles dengan 'Blank'
        df_clean['Role'] = df_clean['Role'].fillna('Blank')
        
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

    def match_complaint_tickets(self, raw_df, complaint_df):
        """Match tickets antara raw data dan complaint data berdasarkan phone number"""
        complaint_tickets = {}
        
        if complaint_df is None or 'Cleaned_Phone' not in complaint_df.columns:
            print("⚠️ No complaint data or phone column not found")
            return complaint_tickets
        
        # Extract phones dari raw data - PERBAIKAN: gunakan raw_df yang benar
        raw_phones = self._extract_phones_from_raw_data(raw_df)
        
        for _, complaint_row in complaint_df.iterrows():
            complaint_phone = complaint_row['Cleaned_Phone']
            
            if pd.isna(complaint_phone) or complaint_phone == 'nan':
                continue
                
            # Cari matching phone di raw data
            matching_tickets = []
            for ticket_id, phone_info in raw_phones.items():
                if phone_info['phone'] and complaint_phone in phone_info['phone']:
                    matching_tickets.append(ticket_id)
            
            if matching_tickets:
                complaint_tickets[complaint_phone] = {
                    'ticket_numbers': matching_tickets,
                    'lead_time_days': complaint_row.get('Lead Time (Solved)'),
                    'complaint_data': complaint_row.to_dict()
                }
                print(f"✅ Matched phone {complaint_phone} with tickets: {matching_tickets}")
        
        print(f"📊 Found {len(complaint_tickets)} complaint-ticket matches")
        return complaint_tickets
    
    def _extract_phones_from_raw_data(self, df):
        """Extract phone numbers dari raw data"""
        phone_info = {}
        
        for ticket_id in df['Ticket Number'].unique():
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            phone = None
            # Cari phone number di semua kolom teks
            for col in ticket_df.columns:
                if pd.api.types.is_string_dtype(ticket_df[col]):
                    phone_patterns = [r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b']
                    for pattern in phone_patterns:
                        matches = ticket_df[col].astype(str).str.extract(f'({pattern})', expand=False)
                        if not matches.isna().all():
                            phone = matches.dropna().iloc[0] if not matches.dropna().empty else None
                            if phone:
                                phone = re.sub(r'\D', '', phone)  # Clean phone
                                break
                if phone:
                    break
            
            phone_info[ticket_id] = {'phone': phone}
        
        return phone_info

# Conversation Parser dengan Logic Baru
class ConversationParser:
    def __init__(self):
        self.question_indicators = [
            '?', 'apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'kenapa',
            'bisa', 'boleh', 'minta', 'tolong', 'tanya', 'info', 'caranya',
            'mau tanya', 'boleh tanya', 'minta info', 'berapa harga',
            'bagaimana cara', 'bisa tolong', 'mohon bantuan', 'gimana'
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
        
    def detect_conversation_start(self, ticket_df):
        """Deteksi kapan conversation benar-benar dimulai dengan operator"""
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        print(f"   🔍 Analyzing {len(ticket_df)} messages for conversation start...")
        
        # Cari operator greeting message
        for idx, row in ticket_df.iterrows():
            message = str(row['Message']).lower()
            role = str(row['Role']).lower()
            
            if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                for pattern in self.operator_greeting_patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        print(f"   ✅ Conversation start: operator greeting at position {idx}")
                        return row['parsed_timestamp']
        
        # Fallback: first operator message
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            if any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                print(f"   ✅ Conversation start: first operator message at position {idx}")
                return row['parsed_timestamp']
        
        print("   ❌ No conversation start detected")
        return None
    
    def parse_conversation(self, ticket_df):
        """Parse conversation menjadi Q-A pairs dengan logic baru"""
        conversation_start = self.detect_conversation_start(ticket_df)
        
        if not conversation_start:
            print("   ⚠️  No conversation start detected, using all messages")
            conv_df = ticket_df.copy()
        else:
            conv_df = ticket_df[ticket_df['parsed_timestamp'] >= conversation_start]
        
        print(f"   📝 Analyzing {len(conv_df)} messages after conversation start")
        
        if len(conv_df) == 0:
            print("   ❌ No messages after conversation start")
            return []
        
        # Urutkan berdasarkan timestamp
        conv_df = conv_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        qa_pairs = []
        current_question = None
        question_time = None
        
        for idx, row in conv_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            timestamp = row['parsed_timestamp']
            
            # CUSTOMER MESSAGE - potential question
            if any(keyword in role for keyword in ['customer', 'user', 'pelanggan']):
                if self._is_meaningful_question(message):
                    # Jika ada previous question, simpan dulu
                    if current_question:
                        self._save_qa_pair(qa_pairs, current_question, question_time, None, None)
                    
                    # Start new question
                    current_question = message
                    question_time = timestamp
                    print(f"   💬 Customer question: {message[:50]}...")
            
            # OPERATOR MESSAGE - potential answer
            elif current_question and any(keyword in role for keyword in ['operator', 'agent', 'admin', 'cs']):
                answer = message
                
                # Skip generic replies
                if self._is_generic_reply(answer):
                    continue
                
                # Pastikan ini jawaban (setelah question)
                time_gap = (timestamp - question_time).total_seconds()
                if time_gap >= 0:  
                    lead_time = time_gap
                    self._save_qa_pair(qa_pairs, current_question, question_time, answer, timestamp, role, lead_time)
                    print(f"   ✅ Operator answer: {answer[:50]}... (LT: {lead_time/60:.1f}min)")
                    
                    # Reset untuk next question
                    current_question = None
                    question_time = None
        
        # Handle last question jika ada
        if current_question:
            self._save_qa_pair(qa_pairs, current_question, question_time, None, None)
            print(f"   ❓ Unanswered question: {current_question[:50]}...")
        
        # URUTKAN Q-A PAIRS BERDASARKAN QUESTION TIME
        qa_pairs = sorted(qa_pairs, key=lambda x: x['question_time'] if x['question_time'] else pd.Timestamp.min)
        
        print(f"   ✅ Found {len(qa_pairs)} Q-A pairs")
        return qa_pairs
    
    def _is_meaningful_question(self, message):
        """Check jika message meaningful question"""
        if not message or len(message.strip()) < 3:
            return False
            
        message_lower = message.lower().strip()
        
        # Skip very short messages yang cuma greetings
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam']
        words = message_lower.split()
        if len(words) <= 2 and any(word in words for word in greetings):
            return False
        
        # Question indicators
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_question_mark = '?' in message_lower
        
        # Meaningful content check
        meaningful_words = [w for w in words if len(w) > 2 and w not in greetings]
        has_meaningful_content = len(meaningful_words) >= 2
        
        return (has_question_indicator and has_meaningful_content) or has_question_mark or len(meaningful_words) >= 3
    
    def _is_generic_reply(self, message):
        """Skip generic/bot replies"""
        message_lower = str(message).lower()
        generic_patterns = [
            r'virtual\s+assistant',
            r'akan\s+segera\s+menghubungi', 
            r'dalam\s+antrian',
            r'silakan\s+memilih\s+dari\s+menu'
        ]
        return any(re.search(pattern, message_lower) for pattern in generic_patterns)
    
    def _save_qa_pair(self, qa_pairs, question, question_time, answer, answer_time, answer_role=None, lead_time=None):
        """Save Q-A pair ke list"""
        pair_data = {
            'question': question,
            'question_time': question_time,
            'is_answered': answer is not None
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
    
    def _seconds_to_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "00:00:00"

# Main Issue Detector dengan Logic Baru
class MainIssueDetector:
    def __init__(self):
        # Keyword untuk menentukan jenis issue
        self.solution_keywords = [
            'solusi', 'jawaban', 'caranya', 'prosedur', 'bisa menghubungi',
            'silakan menghubungi', 'disarankan untuk', 'rekomendasi'
        ]
        
        self.complaint_keywords = [
            'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan',
            'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa'
        ]
    
    def detect_main_issue(self, qa_pairs):
        """Deteksi main issue dari Q-A pairs - LOGIC BARU"""
        if not qa_pairs:
            return None
        
        # Ambil question pertama sebagai main issue
        main_question = qa_pairs[0]['question']
        main_question_time = qa_pairs[0]['question_time']
        
        return {
            'question': main_question,
            'question_time': main_question_time,
            'qa_pairs': qa_pairs  # Simpan semua QA pairs untuk analysis selanjutnya
        }

# Reply Analyzer dengan Logic Baru
class ReplyAnalyzer:
    def __init__(self, complaint_tickets=None):
        self.complaint_tickets = complaint_tickets or {}
        self.action_keywords = config.ACTION_KEYWORDS
    
    def analyze_replies(self, ticket_id, ticket_df, qa_pairs, main_issue):
        """Analyze replies dengan LOGIC BARU"""
        print(f"🔍 Analyzing replies for ticket {ticket_id}")
        
        # Cek apakah ini complaint ticket
        is_complaint, complaint_data = self._is_complaint_ticket(ticket_id)
        
        if is_complaint:
            print("   🚨 COMPLAINT ticket detected")
            return self._analyze_complaint_replies(ticket_id, ticket_df, qa_pairs, main_issue, complaint_data)
        
        # Cek apakah ada keyword "Ticket Has Been Reopened by"
        has_reopened = self._has_ticket_reopened(ticket_df)
        
        if has_reopened:
            print("   ⚠️  SERIOUS ticket detected (has reopened keyword)")
            return self._analyze_serious_replies(ticket_df, qa_pairs, main_issue)
        else:
            print("   ✅ NORMAL ticket detected")
            return self._analyze_normal_replies(ticket_df, qa_pairs, main_issue)
    
    def _is_complaint_ticket(self, ticket_id):
        """Cek apakah ticket termasuk complaint"""
        for phone, complaint_info in self.complaint_tickets.items():
            if ticket_id in complaint_info['ticket_numbers']:
                return True, complaint_info
        return False, None
    
    def _has_ticket_reopened(self, ticket_df):
        """Cek apakah ada keyword Ticket Has Been Reopened by"""
        for _, row in ticket_df.iterrows():
            if config.TICKET_REOPENED_KEYWORD in str(row['Message']):
                return True
        return False
    
    def _analyze_complaint_replies(self, ticket_id, ticket_df, qa_pairs, main_issue, complaint_data):
        """Analyze replies untuk complaint tickets"""
        # Cari first reply (operator reply pertama setelah main question)
        first_reply = self._find_first_reply(ticket_df, main_issue['question_time'])
        
        first_reply_found = first_reply is not None
        final_reply_found = True  # Untuk complaint, selalu dianggap ada final reply dari system
        
        # PERBAIKAN: Pass reply status ke customer leave check
        customer_leave = self._check_customer_leave(ticket_df, first_reply_found, final_reply_found)
        
        analysis_result = {
            'issue_type': 'complaint',
            'first_reply': first_reply,
            'final_reply': {
                'message': 'COMPLAINT_RESOLUTION',
                'timestamp': None,
                'lead_time_minutes': None,
                'lead_time_days': complaint_data.get('lead_time_days'),
                'note': 'Final resolution from complaint system'
            },
            'customer_leave': customer_leave,
            'requirement_compliant': first_reply is not None
        }
        
        return analysis_result
    
    def _analyze_serious_replies(self, ticket_df, qa_pairs, main_issue):
        """Analyze replies untuk serious tickets"""
        # Cari timestamp ticket reopened
        reopened_time = self._find_ticket_reopened_time(ticket_df)
        
        # First reply: operator reply SEBELUM ticket reopened yang mengandung action
        first_reply = self._find_serious_first_reply(ticket_df, main_issue['question_time'], reopened_time)
        
        # Final reply: operator reply PERTAMA SETELAH ticket reopened
        final_reply = self._find_serious_final_reply(ticket_df, reopened_time)
        
        first_reply_found = first_reply is not None
        final_reply_found = final_reply is not None
        
        # PERBAIKAN: Pass reply status ke customer leave check
        customer_leave = self._check_customer_leave(ticket_df, first_reply_found, final_reply_found)
        
        analysis_result = {
            'issue_type': 'serious',
            'first_reply': first_reply,
            'final_reply': final_reply,
            'customer_leave': customer_leave,
            'requirement_compliant': first_reply is not None and final_reply is not None
        }
        
        return analysis_result
    
    def _analyze_normal_replies(self, ticket_df, qa_pairs, main_issue):
        """Analyze replies untuk normal tickets"""
        # Cari operator reply yang mengandung solusi (langsung dianggap final reply)
        final_reply = self._find_normal_final_reply(ticket_df, main_issue['question_time'])
        
        first_reply_found = False  # Normal tidak butuh first reply
        final_reply_found = final_reply is not None
        
        # PERBAIKAN: Pass reply status ke customer leave check
        customer_leave = self._check_customer_leave(ticket_df, first_reply_found, final_reply_found)
        
        analysis_result = {
            'issue_type': 'normal',
            'first_reply': None,  # Normal tidak butuh first reply
            'final_reply': final_reply,
            'customer_leave': customer_leave,
            'requirement_compliant': final_reply is not None
        }
        
        return analysis_result
    
    def _find_first_reply(self, ticket_df, question_time):
        """Cari first reply (operator reply pertama setelah question)"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        if not operator_messages.empty:
            first_msg = operator_messages.iloc[0]
            lead_time = (first_msg['parsed_timestamp'] - question_time).total_seconds()
            
            return {
                'message': first_msg['Message'],
                'timestamp': first_msg['parsed_timestamp'],
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2),
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time)
            }
        return None
    
    def _find_serious_first_reply(self, ticket_df, question_time, reopened_time):
        """Cari serious first reply (sebelum reopened, mengandung action)"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['parsed_timestamp'] < reopened_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        for _, msg in operator_messages.iterrows():
            if self._contains_action_keyword(msg['Message']):
                lead_time = (msg['parsed_timestamp'] - question_time).total_seconds()
                
                return {
                    'message': msg['Message'],
                    'timestamp': msg['parsed_timestamp'],
                    'lead_time_seconds': lead_time,
                    'lead_time_minutes': round(lead_time / 60, 2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                    'note': 'Contains action keyword'
                }
        
        # Fallback: ambil operator reply pertama sebelum reopened
        if not operator_messages.empty:
            first_msg = operator_messages.iloc[0]
            lead_time = (first_msg['parsed_timestamp'] - question_time).total_seconds()
            
            return {
                'message': first_msg['Message'],
                'timestamp': first_msg['parsed_timestamp'],
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2),
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                'note': 'First operator reply before reopened'
            }
        
        return None
    
    def _find_serious_final_reply(self, ticket_df, reopened_time):
        """Cari serious final reply (pertama setelah reopened)"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > reopened_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        if not operator_messages.empty:
            first_msg = operator_messages.iloc[0]
            
            return {
                'message': first_msg['Message'],
                'timestamp': first_msg['parsed_timestamp'],
                'note': 'First operator reply after ticket reopened'
            }
        return None
    
    def _find_normal_final_reply(self, ticket_df, question_time):
        """Cari normal final reply (mengandung solusi) - PERBAIKAN: skip generic replies"""
        operator_messages = ticket_df[
            (ticket_df['parsed_timestamp'] > question_time) &
            (ticket_df['Role'].str.lower().str.contains('operator|agent|admin|cs', na=False))
        ].sort_values('parsed_timestamp')
        
        # Skip patterns untuk reply yang tidak mengandung solusi
        non_solution_patterns = [
            'apabila sudah cukup',
            'apakah sudah cukup', 
            'apakah informasinya sudah cukup',
            'terima kasih telah menghubungi',
            'selamat beraktivitas',
            'goodbye',
            'bye',
            'sampai jumpa'
        ]
        
        # Cari yang mengandung solusi dan BUKAN generic reply
        for _, msg in operator_messages.iterrows():
            message = str(msg['Message']).lower()
            
            # Skip jika mengandung pattern non-solution
            if any(pattern in message for pattern in non_solution_patterns):
                continue
                
            if self._contains_solution_keyword(msg['Message']):
                lead_time = (msg['parsed_timestamp'] - question_time).total_seconds()
                
                return {
                    'message': msg['Message'],
                    'timestamp': msg['parsed_timestamp'],
                    'lead_time_seconds': lead_time,
                    'lead_time_minutes': round(lead_time / 60, 2),
                    'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                    'note': 'Contains solution'
                }
        
        # Fallback: ambil operator reply pertama yang BUKAN generic
        for _, msg in operator_messages.iterrows():
            message = str(msg['Message']).lower()
            
            # Skip generic replies
            if any(pattern in message for pattern in non_solution_patterns):
                continue
                
            lead_time = (msg['parsed_timestamp'] - question_time).total_seconds()
            
            return {
                'message': msg['Message'],
                'timestamp': msg['parsed_timestamp'],
                'lead_time_seconds': lead_time,
                'lead_time_minutes': round(lead_time / 60, 2),
                'lead_time_hhmmss': self._seconds_to_hhmmss(lead_time),
                'note': 'First non-generic operator reply'
            }
        
        return None
    
    def _find_ticket_reopened_time(self, ticket_df):
        """Cari timestamp ketika ticket di-reopen"""
        for _, row in ticket_df.iterrows():
            if config.TICKET_REOPENED_KEYWORD in str(row['Message']):
                return row['parsed_timestamp']
        return None
    
    def _contains_action_keyword(self, message):
        """Cek apakah message mengandung action keyword"""
        message_lower = str(message).lower()
        return any(keyword in message_lower for keyword in self.action_keywords)
    
    def _contains_solution_keyword(self, message):
        """Cek apakah message mengandung solution keyword"""
        message_lower = str(message).lower()
        solution_keywords = [
            'solusi', 'jawaban', 'caranya', 'prosedur', 'bisa menghubungi',
            'silakan menghubungi', 'disarankan untuk', 'rekomendasi'
        ]
        return any(keyword in message_lower for keyword in solution_keywords)
    
    def _check_customer_leave(self, ticket_df, first_reply_found, final_reply_found):
        """Cek apakah customer leave conversation - LOGIC BARU"""
        # Cari keyword customer leave dari ticket automation
        has_leave_keyword = False
        for _, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            if 'ticket automation' in role and config.CUSTOMER_LEAVE_KEYWORD in message:
                has_leave_keyword = True
                break
        
        # PERBAIKAN: Hanya dianggap customer leave jika:
        # 1. Ada keyword customer leave DAN
        # 2. Tidak ada first reply ATAU tidak ada final reply
        if has_leave_keyword and (not first_reply_found or not final_reply_found):
            print("   🚶 Customer leave detected (no proper replies)")
            return True
        elif has_leave_keyword and first_reply_found and final_reply_found:
            print("   ⚠️  Customer leave keyword found but replies exist - NOT counted as leave")
            return False
        else:
            return False
    
    def _seconds_to_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "00:00:00"

class CompleteAnalysisPipeline:
    def __init__(self, complaint_data_path=None):
        self.preprocessor = DataPreprocessor()
        self.parser = ConversationParser()
        self.issue_detector = MainIssueDetector()
        self.complaint_tickets = {}
        self.complaint_data_path = complaint_data_path
        
        print("🚀 Complete Analysis Pipeline Initialized")
    
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
    
    def analyze_all_tickets(self, df, max_tickets=None):
        """Analisis semua tickets dengan complaint matching yang benar"""
        print("🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        
        # Load complaint data setelah punya raw_df
        if self.complaint_data_path and os.path.exists(self.complaint_data_path):
            complaint_df = self.preprocessor.load_complaint_data(self.complaint_data_path)
            self.complaint_tickets = self.preprocessor.match_complaint_tickets(df, complaint_df)
        
        # Initialize reply analyzer dengan complaint_tickets yang sudah di-load
        self.reply_analyzer = ReplyAnalyzer(self.complaint_tickets)
        
        ticket_ids = df['Ticket Number'].unique()
        
        if max_tickets:
            ticket_ids = ticket_ids[:max_tickets]
            print(f"🔍 Analyzing {max_tickets} tickets (max limit)...")
        else:
            print(f"🔍 Analyzing {len(ticket_ids)} tickets...")
        
        self.results = []
        successful_analyses = 0
        
        for i, ticket_id in enumerate(ticket_ids):
            ticket_df = df[df['Ticket Number'] == ticket_id]
            
            result = self.analyze_single_ticket(ticket_df, ticket_id)
            self.results.append(result)
            
            if result['status'] == 'success':
                successful_analyses += 1
            
            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(ticket_ids):
                progress = (i + 1) / len(ticket_ids) * 100
                print(f"   📊 Progress: {i + 1}/{len(ticket_ids)} ({progress:.1f}%) - {successful_analyses} successful")
        
        # Calculate statistics
        self.analysis_stats = self._calculate_stats(len(ticket_ids))
        
        print(f"\n🎉 ANALYSIS PIPELINE COMPLETED!")
        self._print_summary_report()
        
        return self.results, self.analysis_stats
    
    def analyze_single_ticket(self, ticket_df, ticket_id):
        """Analisis lengkap untuk single ticket"""
        print(f"🎯 Analyzing Ticket: {ticket_id}")
        
        try:
            # Step 1: Parse Q-A pairs
            qa_pairs = self.parser.parse_conversation(ticket_df)
            
            if not qa_pairs:
                return self._create_ticket_result(ticket_id, "failed", "No Q-A pairs detected", {})
            
            print(f"   ✓ Found {len(qa_pairs)} Q-A pairs")
            
            # Step 2: Detect main issue
            main_issue = self.issue_detector.detect_main_issue(qa_pairs)
            
            if not main_issue:
                return self._create_ticket_result(ticket_id, "failed", "No main issue detected", {})
            
            print(f"   ✓ Main issue detected: {main_issue['question'][:50]}...")
            
            # Step 3: Analyze replies dengan LOGIC BARU
            if not hasattr(self, 'reply_analyzer'):
                self.reply_analyzer = ReplyAnalyzer(self.complaint_tickets)
                
            reply_analysis = self.reply_analyzer.analyze_replies(ticket_id, ticket_df, qa_pairs, main_issue)
            
            print(f"   ✓ Reply analysis: {reply_analysis['issue_type']}")
            
            # Step 4: Compile results
            result = self._compile_ticket_result(
                ticket_id, ticket_df, qa_pairs, main_issue, reply_analysis
            )
            
            print(f"   ✅ Analysis completed")
            return result
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"   ❌ Analysis failed: {error_msg}")
            return self._create_ticket_result(ticket_id, "failed", error_msg, {})
    
    def _compile_ticket_result(self, ticket_id, ticket_df, qa_pairs, main_issue, reply_analysis):
        """Compile ticket result - FIXED: Ensure numeric lead times"""
        # Hitung quality score
        quality_score = 0
        if reply_analysis['first_reply']:
            quality_score += 2
        if reply_analysis['final_reply']:
            quality_score += 2
        if not reply_analysis['customer_leave']:
            quality_score += 1
        
        # Tentukan performance rating
        if reply_analysis['requirement_compliant']:
            performance_rating = 'good'
        else:
            performance_rating = 'fair'
        
        # FIXED: Ensure lead times are numeric or None
        first_lead_minutes = None
        final_lead_minutes = None
        
        if reply_analysis['first_reply'] and reply_analysis['first_reply'].get('lead_time_minutes'):
            try:
                first_lead_minutes = float(reply_analysis['first_reply']['lead_time_minutes'])
            except (ValueError, TypeError):
                first_lead_minutes = None
        
        if reply_analysis['final_reply'] and reply_analysis['final_reply'].get('lead_time_minutes'):
            try:
                final_lead_minutes = float(reply_analysis['final_reply']['lead_time_minutes'])
            except (ValueError, TypeError):
                final_lead_minutes = None
        
        result = {
            'ticket_id': ticket_id,
            'status': 'success',
            'analysis_timestamp': datetime.now(),
            
            # Conversation info
            'total_messages': len(ticket_df),
            'total_qa_pairs': len(qa_pairs),
            'answered_pairs': sum(1 for pair in qa_pairs if pair['is_answered']),
            'customer_leave': reply_analysis['customer_leave'],
            
            # Main issue analysis
            'main_question': main_issue['question'],
            'main_question_time': main_issue['question_time'],
            'final_issue_type': reply_analysis['issue_type'],
            
            # Reply analysis
            'first_reply_found': reply_analysis['first_reply'] is not None,
            'final_reply_found': reply_analysis['final_reply'] is not None,
            'first_reply_message': reply_analysis['first_reply']['message'] if reply_analysis['first_reply'] else None,
            'first_reply_time': reply_analysis['first_reply']['timestamp'] if reply_analysis['first_reply'] else None,
            'final_reply_message': reply_analysis['final_reply']['message'] if reply_analysis['final_reply'] else None,
            'final_reply_time': reply_analysis['final_reply']['timestamp'] if reply_analysis['final_reply'] else None,
            
            # Lead times - FIXED: Use numeric values
            'first_reply_lead_time_minutes': first_lead_minutes,
            'final_reply_lead_time_minutes': final_lead_minutes,
            'first_reply_lead_time_hhmmss': reply_analysis['first_reply'].get('lead_time_hhmmss') if reply_analysis['first_reply'] else None,
            'final_reply_lead_time_hhmmss': reply_analysis['final_reply'].get('lead_time_hhmmss') if reply_analysis['final_reply'] else None,
            
            # Untuk complaint
            'final_reply_lead_time_days': reply_analysis['final_reply'].get('lead_time_days') if reply_analysis['final_reply'] else None,
            
            # Performance metrics
            'performance_rating': performance_rating,
            'quality_score': quality_score,
            'quality_rating': 'good' if quality_score >= 4 else 'fair' if quality_score >= 2 else 'poor',
            'requirement_compliant': reply_analysis['requirement_compliant'],
            
            # Raw data
            '_raw_qa_pairs': qa_pairs,
            '_raw_reply_analysis': reply_analysis
        }
        
        return result
    
    def _calculate_stats(self, total_tickets):
        """Hitung statistics dari results - FIXED VERSION"""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        stats = {
            'total_tickets': len(self.results),
            'successful_analysis': len(successful),
            'failed_analysis': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0
        }
        
        if successful:
            # Issue type distribution
            issue_types = [r['final_issue_type'] for r in successful]
            stats['issue_type_distribution'] = dict(Counter(issue_types))
            
            # Performance metrics
            performance_ratings = [r['performance_rating'] for r in successful]
            stats['performance_distribution'] = dict(Counter(performance_ratings))
            
            # Lead time statistics - FIXED: Selalu include semua key yang diperlukan
            first_lead_times = []
            final_lead_times = []
            
            for r in successful:
                # First reply lead times
                first_lt = r.get('first_reply_lead_time_minutes')
                if first_lt is not None and first_lt != 'N/A':
                    try:
                        first_lead_times.append(float(first_lt))
                    except (ValueError, TypeError):
                        pass
                
                # Final reply lead times (hanya untuk normal/serious, bukan complaint)
                final_lt = r.get('final_reply_lead_time_minutes')
                if final_lt is not None and final_lt != 'N/A':
                    try:
                        final_lead_times.append(float(final_lt))
                    except (ValueError, TypeError):
                        pass
            
            # PERBAIKAN: Selalu buat key yang diperlukan meskipun datanya kosong
            stats['lead_time_stats'] = {
                'first_reply_avg_minutes': np.mean(first_lead_times) if first_lead_times else 0,
                'final_reply_avg_minutes': np.mean(final_lead_times) if final_lead_times else 0,
                'first_reply_samples': len(first_lead_times),
                'final_reply_samples': len(final_lead_times)
            }
            
            # Reply effectiveness
            stats['reply_effectiveness'] = {
                'first_reply_found_rate': sum(1 for r in successful if r['first_reply_found']) / len(successful) if successful else 0,
                'final_reply_found_rate': sum(1 for r in successful if r['final_reply_found']) / len(successful) if successful else 0,
                'customer_leave_cases': sum(1 for r in successful if r['customer_leave'])
            }
        
        return stats
    
    def _print_summary_report(self):
        """Print summary report - FIXED VERSION"""
        stats = self.analysis_stats
        
        print(f"📊 ANALYSIS SUMMARY REPORT")
        print(f"   • Total Tickets: {stats['total_tickets']}")
        print(f"   • Successful Analysis: {stats['successful_analysis']} ({stats['success_rate']*100:.1f}%)")
        
        if 'issue_type_distribution' in stats:
            print(f"   • Issue Types: {stats['issue_type_distribution']}")
        
        if 'lead_time_stats' in stats:
            lt_stats = stats['lead_time_stats']
            # PERBAIKAN: Gunakan get() dengan default value
            first_reply_avg = lt_stats.get('first_reply_avg_minutes', 0)
            final_reply_avg = lt_stats.get('final_reply_avg_minutes', 0)
            
            print(f"   • Avg First Reply: {first_reply_avg:.1f} min")
            
            # PERBAIKAN: Handle case dimana final_reply_avg_minutes tidak ada atau 0
            if final_reply_avg > 0 and final_reply_avg != float('inf'):
                print(f"   • Avg Final Reply: {final_reply_avg:.1f} min")
            else:
                print(f"   • Avg Final Reply: Mixed (minutes/days)")
            
            print(f"   • First Reply Samples: {lt_stats.get('first_reply_samples', 0)}")
            print(f"   • Final Reply Samples: {lt_stats.get('final_reply_samples', 0)}")
        
        if 'reply_effectiveness' in stats:
            eff = stats['reply_effectiveness']
            print(f"   • First Reply Found: {eff.get('first_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Final Reply Found: {eff.get('final_reply_found_rate', 0)*100:.1f}%")
            print(f"   • Customer Leave Cases: {eff.get('customer_leave_cases', 0)}")
            
# Results Exporter
class ResultsExporter:
    def __init__(self):
        self.output_dir = "output/"
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def export_comprehensive_results(self, results, stats):
        """Export results ke Excel - PERBAIKAN: lebih lengkap"""
        try:
            output_path = f"{self.output_dir}analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Detailed Results
                detailed_data = []
                for result in results:
                    if result['status'] == 'success':
                        detailed_data.append({
                            'Ticket_ID': result['ticket_id'],
                            'Issue_Type': result['final_issue_type'],
                            'Main_Question': result['main_question'],
                            'Main_Question_Time': result.get('main_question_time'),
                            'First_Reply_Found': result['first_reply_found'],
                            'First_Reply_Message': result.get('first_reply_message', '')[:500] + '...' if result.get('first_reply_message') else 'Not found',
                            'First_Reply_Time': result.get('first_reply_time'),
                            'First_Reply_Lead_Time_Min': result.get('first_reply_lead_time_minutes'),
                            'First_Reply_Lead_Time_Format': result.get('first_reply_lead_time_hhmmss'),
                            'Final_Reply_Found': result['final_reply_found'],
                            'Final_Reply_Message': result.get('final_reply_message', '')[:500] + '...' if result.get('final_reply_message') else 'Not found',
                            'Final_Reply_Time': result.get('final_reply_time'),
                            'Final_Reply_Lead_Time_Min': result.get('final_reply_lead_time_minutes'),
                            'Final_Reply_Lead_Time_Days': result.get('final_reply_lead_time_days'),
                            'Final_Reply_Lead_Time_Format': result.get('final_reply_lead_time_hhmmss'),
                            'Performance_Rating': result['performance_rating'],
                            'Quality_Score': result['quality_score'],
                            'Quality_Rating': result['quality_rating'],
                            'Customer_Leave': result['customer_leave'],
                            'Requirement_Compliant': result['requirement_compliant'],
                            'Total_Messages': result['total_messages'],
                            'Total_QA_Pairs': result['total_qa_pairs'],
                            'Answered_Pairs': result['answered_pairs']
                        })
                    else:
                        detailed_data.append({
                            'Ticket_ID': result['ticket_id'],
                            'Issue_Type': 'FAILED',
                            'Main_Question': result.get('failure_reason', 'Analysis failed'),
                            'First_Reply_Found': False,
                            'Final_Reply_Found': False,
                            'Performance_Rating': 'FAILED',
                            'Quality_Score': 0
                        })
                
                if detailed_data:
                    df_detailed = pd.DataFrame(detailed_data)
                    df_detailed.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Sheet 2: Q-A Pairs Raw Data
                qa_pairs_data = []
                for result in results:
                    if result['status'] == 'success' and '_raw_qa_pairs' in result:
                        for i, qa_pair in enumerate(result['_raw_qa_pairs']):
                            qa_pairs_data.append({
                                'Ticket_ID': result['ticket_id'],
                                'QA_Pair_Index': i + 1,
                                'Question': qa_pair.get('question', ''),
                                'Question_Time': qa_pair.get('question_time'),
                                'Is_Answered': qa_pair.get('is_answered', False),
                                'Answer': qa_pair.get('answer', ''),
                                'Answer_Time': qa_pair.get('answer_time'),
                                'Lead_Time_Minutes': qa_pair.get('lead_time_minutes'),
                                'Lead_Time_Format': qa_pair.get('lead_time_hhmmss')
                            })
                
                if qa_pairs_data:
                    df_qa = pd.DataFrame(qa_pairs_data)
                    df_qa.to_excel(writer, sheet_name='Raw_QA_Pairs', index=False)
                
                # Sheet 3: Summary Statistics
                summary_data = self._create_summary_data(stats)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False, header=False)
                
                # Sheet 4: Performance Metrics
                performance_data = self._create_performance_data(results)
                if performance_data:
                    df_perf = pd.DataFrame(performance_data)
                    df_perf.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            
            print(f"💾 Results exported to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return None
    
    def _create_performance_data(self, results):
        """Create performance metrics data"""
        performance_data = []
        
        for result in results:
            if result['status'] == 'success':
                performance_data.append({
                    'Ticket_ID': result['ticket_id'],
                    'Issue_Type': result['final_issue_type'],
                    'Performance_Rating': result['performance_rating'],
                    'Quality_Score': result['quality_score'],
                    'Quality_Rating': result['quality_rating'],
                    'First_Reply_Found': result['first_reply_found'],
                    'Final_Reply_Found': result['final_reply_found'],
                    'Customer_Leave': result['customer_leave'],
                    'Requirement_Compliant': result['requirement_compliant'],
                    'Total_Messages': result['total_messages'],
                    'Answer_Rate': f"{(result['answered_pairs'] / result['total_qa_pairs']) * 100:.1f}%" if result['total_qa_pairs'] > 0 else '0%'
                })
        
        return performance_data
    
    def _create_summary_data(self, stats):
        """Create summary data untuk Excel"""
        summary_data = [
            ['ENHANCED ANALYSIS SUMMARY REPORT', ''],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['', ''],
            ['OVERALL STATISTICS', ''],
            ['Total Tickets', stats['total_tickets']],
            ['Successful Analysis', stats['successful_analysis']],
            ['Failed Analysis', stats['failed_analysis']],
            ['Success Rate', f"{stats['success_rate']*100:.1f}%"],
            ['', '']
        ]
        
        if 'issue_type_distribution' in stats:
            summary_data.append(['ISSUE TYPE DISTRIBUTION', ''])
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                summary_data.append([f'{issue_type.title()} Issues', f'{count} ({percentage:.1f}%)'])
            summary_data.append(['', ''])
        
        if 'lead_time_stats' in stats:
            lt_stats = stats['lead_time_stats']
            summary_data.append(['LEAD TIME STATISTICS', ''])
            summary_data.append(['First Reply Avg (min)', f"{lt_stats['first_reply_avg_minutes']:.1f}"])
            summary_data.append(['Final Reply Avg (min)', f"{lt_stats['final_reply_avg_minutes']:.1f}"])
            summary_data.append(['First Reply Samples', lt_stats['first_reply_samples']])
            summary_data.append(['Final Reply Samples', lt_stats['final_reply_samples']])
            summary_data.append(['', ''])
        
        if 'reply_effectiveness' in stats:
            eff = stats['reply_effectiveness']
            summary_data.append(['REPLY EFFECTIVENESS', ''])
            summary_data.append(['First Reply Found Rate', f"{eff['first_reply_found_rate']*100:.1f}%"])
            summary_data.append(['Final Reply Found Rate', f"{eff['final_reply_found_rate']*100:.1f}%"])
            summary_data.append(['Customer Leave Cases', eff['customer_leave_cases']])
        
        return summary_data

# Initialize Pipeline
print("✅ ENHANCED Analysis Pipeline Ready!")
print("   ✓ New role handling (Ticket Automation & Blank)")
print("   ✓ New issue type detection logic")
print("   ✓ Complaint ticket matching")
print("   ✓ Ticket reopened detection")
print("=" * 60)
