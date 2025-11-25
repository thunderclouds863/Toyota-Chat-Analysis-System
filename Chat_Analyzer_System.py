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
    
    # Customer leave detection
    CUSTOMER_LEAVE_TIMEOUT = 3  # 3 menit tanpa response dari customer
    
    # Keywords
    TICKET_REOPENED_KEYWORD = "Ticket Has Been Reopened by"
    CUSTOMER_LEAVE_KEYWORD = "Mohon maaf, dikarenakan tidak ada respon, chat ini Kami akhiri. Terima kasih telah menggunakan layanan Live Chat Toyota Astra Motor, selamat beraktivitas kembali."

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
            
            # Validasi columns penting
            required_columns = ['No.Handphone', 'Lead Time (Solved)', 'Ticket Number']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️ Missing columns in complaint data: {missing_columns}")
                return None
            
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

# ===== CONVERSATION PARSER - NEW REQUIREMENT VERSION =====
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
        
        self.action_keywords = [
            'diteruskan', 'disampaikan', 'dihubungi', 'dicek', 'dipelajari',
            'ditindaklanjuti', 'dilakukan pengecekan', 'proses', 'konfirmasi',
            'validasi', 'eskalasi', 'investigasi', 'pengecekan', 'verifikasi',
            'kami lihat', 'kami periksa', 'akan kami', 'tunggu sebentar',
            'mohon ditunggu', 'cek dulu'
        ]
        
        self.solution_keywords = [
            'solusi', 'jawaban', 'caranya', 'prosedur', 'bisa menghubungi',
            'silakan menghubungi', 'disarankan untuk', 'rekomendasi',
            'berikut informasi', 'nomor telepon', 'alamat dealer', 'bengkel resmi',
            'call center', 'hotline', 'customer service', 'info lengkap'
        ]
        
        self.conversation_ender_patterns = [
            'apakah sudah cukup', 'apakah informasinya sudah cukup jelas',
            'terima kasih', 'sampai jumpa', 'goodbye', 'selamat beraktivitas'
        ]

    def find_ticket_reopened_position(self, ticket_df):
        """Cari posisi keyword 'Ticket Has Been Reopened by'"""
        for idx, row in ticket_df.iterrows():
            message = str(row['Message'])
            role = str(row['Role'])
            
            # Cari di role blank atau apapun yang mengandung keyword
            if config.TICKET_REOPENED_KEYWORD in message:
                return idx, row['parsed_timestamp']
        return None, None

    def find_customer_leave_position(self, ticket_df):
        """Cari posisi customer leave keyword"""
        for idx, row in ticket_df.iterrows():
            message = str(row['Message'])
            role = str(row['Role'])
            
            if (role.lower() == 'ticket automation' and 
                config.CUSTOMER_LEAVE_KEYWORD in message):
                return idx, row['parsed_timestamp']
        return None, None

    def detect_customer_leave(self, ticket_df):
        """Deteksi customer leave berdasarkan requirement baru"""
        customer_leave_idx, leave_time = self.find_customer_leave_position(ticket_df)
        
        if not customer_leave_idx:
            return False, None
        
        # Cari last operator message sebelum customer leave
        ticket_df = ticket_df.sort_values('parsed_timestamp')
        leave_df = ticket_df.iloc[:customer_leave_idx]
        
        last_operator_time = None
        for _, row in leave_df[::-1].iterrows():  # Iterate backwards
            if 'operator' in str(row['Role']).lower():
                last_operator_time = row['parsed_timestamp']
                break
        
        if last_operator_time and leave_time:
            time_gap = (leave_time - last_operator_time).total_seconds() / 60
            if time_gap >= config.CUSTOMER_LEAVE_TIMEOUT:
                print(f"   🚶 Customer leave detected: {time_gap:.1f} min gap")
                return True, leave_time
        
        return False, None

    def find_main_question(self, ticket_df):
        """Tentukan main question pertama dari customer"""
        ticket_df = ticket_df.sort_values('parsed_timestamp')
        
        for idx, row in ticket_df.iterrows():
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            # Cari pertama kali customer bertanya meaningful question
            if 'customer' in role and self._is_meaningful_question(message):
                return {
                    'question': message,
                    'timestamp': row['parsed_timestamp'],
                    'position': idx
                }
        
        return None

    def _is_meaningful_question(self, message):
        """Check jika message adalah meaningful question"""
        if not message or len(message.strip()) < 10:
            return False
            
        message_lower = message.lower()
        
        # Skip greetings dan very short messages
        greetings = ['halo', 'hai', 'hi', 'selamat', 'pagi', 'siang', 'sore', 'malam']
        if any(message_lower.startswith(greet) for greet in greetings) and len(message_lower) < 20:
            return False
        
        # Check question indicators
        has_question_indicator = any(indicator in message_lower for indicator in self.question_indicators)
        has_question_mark = '?' in message_lower
        
        return has_question_indicator or has_question_mark or len(message_lower.split()) >= 5

    def parse_conversation(self, ticket_df, complaint_info=None):
        """Parse conversation berdasarkan requirement baru"""
        print(f"   🔍 Parsing conversation with new logic...")
        
        # Urutkan berdasarkan timestamp
        ticket_df = ticket_df.sort_values('parsed_timestamp').reset_index(drop=True)
        
        # Cari main question
        main_question = self.find_main_question(ticket_df)
        if not main_question:
            print("   ❌ No main question found")
            return []
        
        print(f"   ✅ Main question: {main_question['question'][:50]}...")
        
        # Cari ticket reopened position
        reopened_idx, reopened_time = self.find_ticket_reopened_position(ticket_df)
        
        # Deteksi customer leave
        customer_leave, leave_time = self.detect_customer_leave(ticket_df)
        
        # Tentukan issue type
        issue_type = self.determine_issue_type(ticket_df, main_question, reopened_idx, complaint_info)
        
        # Cari replies berdasarkan issue type
        first_reply, final_reply = self.find_replies(ticket_df, main_question, issue_type, reopened_idx, complaint_info)
        
        # Compile Q-A pair
        qa_pair = {
            'question': main_question['question'],
            'question_time': main_question['timestamp'],
            'is_answered': first_reply is not None or final_reply is not None,
            'issue_type': issue_type,
            'customer_leave': customer_leave,
            'reopened_position': reopened_idx,
            'ticket_reopened_time': reopened_time
        }
        
        if first_reply:
            lead_time_seconds = (first_reply['timestamp'] - main_question['timestamp']).total_seconds()
            qa_pair.update({
                'first_reply': first_reply['message'],
                'first_reply_time': first_reply['timestamp'],
                'first_reply_role': first_reply['role'],
                'first_lead_time_seconds': lead_time_seconds,
                'first_lead_time_minutes': round(lead_time_seconds / 60, 2),
                'first_lead_time_hhmmss': self._seconds_to_hhmmss(lead_time_seconds)
            })
        
        if final_reply:
            lead_time_seconds = (final_reply['timestamp'] - main_question['timestamp']).total_seconds()
            qa_pair.update({
                'final_reply': final_reply['message'],
                'final_reply_time': final_reply['timestamp'],
                'final_reply_role': final_reply['role'],
                'final_lead_time_seconds': lead_time_seconds,
                'final_lead_time_minutes': round(lead_time_seconds / 60, 2),
                'final_lead_time_hhmmss': self._seconds_to_hhmmss(lead_time_seconds)
            })
        
        # Untuk complaint, gunakan lead time dari file complaint
        if issue_type == 'complaint' and complaint_info and 'lead_time_days' in complaint_info:
            qa_pair['final_lead_time_days'] = complaint_info['lead_time_days']
            qa_pair['final_lead_time_minutes'] = complaint_info['lead_time_days'] * 24 * 60  # Convert to minutes
        
        return [qa_pair]

    def determine_issue_type(self, ticket_df, main_question, reopened_idx, complaint_info):
        """Tentukan issue type berdasarkan requirement baru"""
        
        # 1. Check jika ini complaint
        if complaint_info:
            return 'complaint'
        
        # 2. Check jika ada ticket reopened (serious)
        if reopened_idx is not None:
            return 'serious'
        
        # 3. Normal inquiry (bisa langsung dijawab tanpa jeda ticket reopened)
        return 'normal'

    def find_replies(self, ticket_df, main_question, issue_type, reopened_idx, complaint_info):
        """Cari first dan final reply berdasarkan issue type"""
        first_reply = None
        final_reply = None
        
        main_question_time = main_question['timestamp']
        main_question_pos = main_question['position']
        
        if issue_type == 'normal':
            # Untuk normal: cari jawaban langsung setelah main question
            first_reply = self._find_direct_reply(ticket_df, main_question_pos, main_question_time)
            final_reply = first_reply  # Untuk normal, first reply = final reply
            
        elif issue_type == 'serious':
            # Untuk serious: first reply sebelum ticket reopened, final reply setelah ticket reopened
            first_reply = self._find_serious_first_reply(ticket_df, main_question_pos, reopened_idx)
            final_reply = self._find_serious_final_reply(ticket_df, reopened_idx)
            
        elif issue_type == 'complaint':
            # Untuk complaint: first reply dari conversation, final reply dari file complaint
            first_reply = self._find_complaint_first_reply(ticket_df, main_question_pos)
            # Final reply lead time dari file complaint, tidak ada message
        
        return first_reply, final_reply

    def _find_direct_reply(self, ticket_df, main_question_pos, main_question_time):
        """Cari direct reply untuk normal inquiry"""
        for idx in range(main_question_pos + 1, len(ticket_df)):
            row = ticket_df.iloc[idx]
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            if 'operator' in role and self._contains_solution(message):
                return {
                    'message': message,
                    'timestamp': row['parsed_timestamp'],
                    'role': row['Role']
                }
        
        return None

    def _find_serious_first_reply(self, ticket_df, main_question_pos, reopened_idx):
        """Cari first reply untuk serious issue (sebelum ticket reopened)"""
        if reopened_idx is None:
            return None
            
        for idx in range(main_question_pos + 1, reopened_idx):
            row = ticket_df.iloc[idx]
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            if 'operator' in role and self._contains_action(message):
                return {
                    'message': message,
                    'timestamp': row['parsed_timestamp'],
                    'role': row['Role']
                }
        
        return None

    def _find_serious_final_reply(self, ticket_df, reopened_idx):
        """Cari final reply untuk serious issue (setelah ticket reopened)"""
        if reopened_idx is None:
            return None
            
        for idx in range(reopened_idx + 1, len(ticket_df)):
            row = ticket_df.iloc[idx]
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            if 'operator' in role:
                return {
                    'message': message,
                    'timestamp': row['parsed_timestamp'],
                    'role': row['Role']
                }
        
        return None

    def _find_complaint_first_reply(self, ticket_df, main_question_pos):
        """Cari first reply untuk complaint"""
        for idx in range(main_question_pos + 1, len(ticket_df)):
            row = ticket_df.iloc[idx]
            role = str(row['Role']).lower()
            message = str(row['Message'])
            
            if 'operator' in role:
                return {
                    'message': message,
                    'timestamp': row['parsed_timestamp'],
                    'role': row['Role']
                }
        
        return None

    def _contains_action(self, message):
        """Check jika message mengandung action keywords"""
        message_lower = message.lower()
        return any(action in message_lower for action in self.action_keywords)

    def _contains_solution(self, message):
        """Check jika message mengandung solution"""
        message_lower = message.lower()
        return any(solution in message_lower for solution in self.solution_keywords)

    def _seconds_to_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except:
            return "00:00:00"

# Complaint Matcher
class ComplaintMatcher:
    def __init__(self):
        self.complaint_data = None
    
    def load_complaint_data(self, file_path):
        """Load complaint data"""
        try:
            self.complaint_data = pd.read_excel(file_path)
            print(f"✅ Loaded {len(self.complaint_data)} complaint records")
            return True
        except Exception as e:
            print(f"❌ Error loading complaint data: {e}")
            return False
    
    def extract_phone_from_conversation(self, ticket_df):
        """Extract phone number dari conversation"""
        for _, row in ticket_df.iterrows():
            message = str(row['Message'])
            
            # Cari phone patterns
            phone_patterns = [
                r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\b\d{2}[-.\s]?\d{4}[-.\s]?\d{4}\b'
            ]
            
            for pattern in phone_patterns:
                matches = re.findall(pattern, message)
                if matches:
                    return matches[0]  # Return first match
        
        return None
    
    def match_complaint(self, ticket_df, ticket_id):
        """Match conversation dengan complaint data"""
        if self.complaint_data is None:
            return None
        
        # Cari phone number dari conversation
        phone = self.extract_phone_from_conversation(ticket_df)
        
        if not phone:
            return None
        
        print(f"   📞 Found phone in conversation: {phone}")
        
        # Cari di complaint data
        for _, complaint_row in self.complaint_data.iterrows():
            complaint_phone = str(complaint_row['No.Handphone'])
            
            # Normalize phone numbers untuk matching
            normalized_phone = self.normalize_phone(phone)
            normalized_complaint_phone = self.normalize_phone(complaint_phone)
            
            if normalized_phone and normalized_complaint_phone and normalized_phone in normalized_complaint_phone:
                print(f"   ✅ Matched complaint for phone: {phone}")
                
                return {
                    'complaint_phone': complaint_phone,
                    'lead_time_days': complaint_row['Lead Time (Solved)'],
                    'ticket_number': complaint_row.get('Ticket Number', 'Unknown'),
                    'complaint_data': dict(complaint_row)
                }
        
        return None
    
    def normalize_phone(self, phone):
        """Normalize phone number untuk matching"""
        if pd.isna(phone):
            return None
        
        phone_str = str(phone)
        # Hanya ambil digits
        digits = re.sub(r'\D', '', phone_str)
        
        # Handle berbagai format
        if digits.startswith('0'):
            digits = '62' + digits[1:]
        elif digits.startswith('8'):
            digits = '62' + digits
        elif digits.startswith('+'):
            digits = digits[1:]
        
        return digits

# Main Issue Detector (Simplified)
class MainIssueDetector:
    def __init__(self):
        # Keyword patterns untuk menentukan issue type
        self.serious_indicators = [
            'error', 'rusak', 'masalah', 'gagal', 'mogok', 'mati', 'tidak bisa',
            'help', 'urgent', 'kendala', 'trouble', 'macet', 'hang', 'blank',
            'not responding', 'bermasalah', 'gangguan'
        ]
        
        self.complaint_indicators = [
            'komplain', 'kecewa', 'marah', 'protes', 'pengaduan', 'keluhan',
            'sakit hati', 'tidak puas', 'keberatan', 'sangat kecewa'
        ]
    
    def detect_issue_type(self, qa_pairs, complaint_info=None):
        """Deteksi issue type - simplified version"""
        if not qa_pairs:
            return 'unknown'
        
        # Priority: complaint > serious > normal
        if complaint_info:
            return 'complaint'
        
        main_question = qa_pairs[0]['question'].lower()
        
        # Check serious indicators
        if any(indicator in main_question for indicator in self.serious_indicators):
            return 'serious'
        
        # Check complaint indicators  
        if any(indicator in main_question for indicator in self.complaint_indicators):
            return 'complaint'
        
        return 'normal'

# Reply Analyzer (Simplified)
class ReplyAnalyzer:
    def __init__(self):
        self.action_patterns = [
            r'diteruskan', r'disampaikan', r'dihubungi', r'dicek', r'dipelajari',
            r'ditindaklanjuti', r'dilakukan pengecekan', r'proses', r'konfirmasi',
            r'validasi', r'eskalasi', r'investigasi', r'pengecekan', r'verifikasi',
            r'kami lihat', r'kami periksa', r'akan kami', r'tunggu sebentar',
            r'mohon ditunggu', r'cek dulu'
        ]
        
        self.solution_patterns = [
            r'solusi', r'jawaban', r'caranya', r'prosedur', r'bisa menghubungi',
            r'silakan menghubungi', r'disarankan untuk', r'rekomendasi',
            r'berikut informasi', r'nomor telepon', r'alamat dealer', r'bengkel resmi',
            r'call center', r'hotline', r'customer service', r'info lengkap'
        ]

    def analyze_replies(self, qa_pairs):
        """Analyze replies dengan logic baru"""
        if not qa_pairs:
            return None, None, {'requirement_compliant': False}
        
        qa_pair = qa_pairs[0]
        issue_type = qa_pair.get('issue_type', 'normal')
        
        first_reply_found = 'first_reply' in qa_pair
        final_reply_found = 'final_reply' in qa_pair
        
        # Validasi requirements
        requirement_compliant = self._validate_requirements(
            issue_type, first_reply_found, final_reply_found, qa_pair.get('customer_leave', False)
        )
        
        analysis_result = {
            'issue_type': issue_type,
            'first_reply_found': first_reply_found,
            'final_reply_found': final_reply_found,
            'customer_leave': qa_pair.get('customer_leave', False),
            'requirement_compliant': requirement_compliant,
            'lead_times': {
                'first_reply_lead_time_minutes': qa_pair.get('first_lead_time_minutes'),
                'final_reply_lead_time_minutes': qa_pair.get('final_lead_time_minutes'),
                'first_reply_lead_time_hhmmss': qa_pair.get('first_lead_time_hhmmss'),
                'final_reply_lead_time_hhmmss': qa_pair.get('final_lead_time_hhmmss')
            }
        }
        
        # Create reply objects
        first_reply = None
        if first_reply_found:
            first_reply = {
                'message': qa_pair['first_reply'],
                'timestamp': qa_pair['first_reply_time'],
                'role': qa_pair.get('first_reply_role', 'Operator'),
                'lead_time_minutes': qa_pair.get('first_lead_time_minutes'),
                'lead_time_hhmmss': qa_pair.get('first_lead_time_hhmmss')
            }
        
        final_reply = None
        if final_reply_found:
            final_reply = {
                'message': qa_pair['final_reply'],
                'timestamp': qa_pair['final_reply_time'],
                'role': qa_pair.get('final_reply_role', 'Operator'),
                'lead_time_minutes': qa_pair.get('final_lead_time_minutes'),
                'lead_time_hhmmss': qa_pair.get('final_lead_time_hhmmss')
            }
        elif issue_type == 'complaint' and 'final_lead_time_days' in qa_pair:
            # Untuk complaint, buat final reply object tanpa message
            final_reply = {
                'message': 'COMPLAINT_RESOLVED',
                'timestamp': None,
                'role': 'System',
                'lead_time_days': qa_pair['final_lead_time_days'],
                'lead_time_minutes': qa_pair.get('final_lead_time_minutes'),
                'note': f"Resolved in {qa_pair['final_lead_time_days']} days (from complaint data)"
            }
        
        return first_reply, final_reply, analysis_result

    def _validate_requirements(self, issue_type, first_reply_found, final_reply_found, customer_leave):
        """Validasi requirements berdasarkan issue type"""
        if issue_type == 'normal':
            return final_reply_found or customer_leave
        elif issue_type == 'serious':
            return first_reply_found  # Final reply optional untuk serious
        elif issue_type == 'complaint':
            return first_reply_found  # Final reply dari file complaint
        return False

# Complete Analysis Pipeline (NEW VERSION)
class CompleteAnalysisPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.parser = ConversationParser()
        self.complaint_matcher = ComplaintMatcher()
        self.issue_detector = MainIssueDetector()
        self.reply_analyzer = ReplyAnalyzer()
        self.results = []
        self.analysis_stats = {}
        
        print("🚀 Complete Analysis Pipeline Initialized (NEW REQUIREMENTS)")

    def load_complaint_data(self, complaint_file_path):
        """Load complaint data"""
        return self.complaint_matcher.load_complaint_data(complaint_file_path)

    def analyze_single_ticket(self, ticket_df, ticket_id):
        """Analisis single ticket dengan logic baru"""
        print(f"🎯 Analyzing Ticket: {ticket_id}")
        
        try:
            # Match dengan complaint data
            complaint_info = self.complaint_matcher.match_complaint(ticket_df, ticket_id)
            
            # Parse conversation dengan logic baru
            qa_pairs = self.parser.parse_conversation(ticket_df, complaint_info)
            
            if not qa_pairs:
                return self._create_ticket_result(ticket_id, "failed", "No Q-A pairs detected", {})
            
            # Analyze replies
            first_reply, final_reply, reply_analysis = self.reply_analyzer.analyze_replies(qa_pairs)
            
            # Compile result
            result = self._compile_ticket_result(
                ticket_id, ticket_df, qa_pairs[0], first_reply, final_reply, reply_analysis, complaint_info
            )
            
            print(f"   ✅ Analysis completed - {result['final_issue_type'].upper()}")
            return result
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"   ❌ Analysis failed: {error_msg}")
            return self._create_ticket_result(ticket_id, "failed", error_msg, {})

    def _compile_ticket_result(self, ticket_id, ticket_df, qa_pair, first_reply, final_reply, reply_analysis, complaint_info):
        """Compile ticket result"""
        
        # Determine performance rating
        if reply_analysis['requirement_compliant']:
            performance_rating = 'good'
        else:
            performance_rating = 'fair'
        
        # Calculate quality score
        quality_score = 0
        if first_reply:
            quality_score += 2
        if final_reply:
            quality_score += 2
        if reply_analysis.get('customer_leave'):
            quality_score += 1
        
        result = {
            'ticket_id': ticket_id,
            'status': 'success',
            'analysis_timestamp': datetime.now(),
            
            # Conversation info
            'total_messages': len(ticket_df),
            'total_qa_pairs': 1,  # Selalu 1 dengan logic baru
            'answered_pairs': 1 if qa_pair['is_answered'] else 0,
            'customer_leave': qa_pair.get('customer_leave', False),
            
            # Main issue
            'main_question': qa_pair['question'],
            'main_question_time': qa_pair['question_time'],
            'final_issue_type': qa_pair['issue_type'],
            'is_complaint': complaint_info is not None,
            
            # Reply analysis
            'first_reply_found': first_reply is not None,
            'final_reply_found': final_reply is not None,
            'first_reply_message': first_reply['message'] if first_reply else None,
            'first_reply_time': first_reply['timestamp'] if first_reply else None,
            'final_reply_message': final_reply['message'] if final_reply else None,
            'final_reply_time': final_reply['timestamp'] if final_reply else None,
            
            # Lead times
            'first_reply_lead_time_minutes': first_reply['lead_time_minutes'] if first_reply else None,
            'final_reply_lead_time_minutes': final_reply['lead_time_minutes'] if final_reply else None,
            'first_reply_lead_time_hhmmss': first_reply['lead_time_hhmmss'] if first_reply else None,
            'final_reply_lead_time_hhmmss': final_reply['lead_time_hhmmss'] if final_reply else None,
            
            # Performance metrics
            'performance_rating': performance_rating,
            'quality_score': quality_score,
            'quality_rating': 'good' if quality_score >= 3 else 'fair',
            'requirement_compliant': reply_analysis['requirement_compliant'],
            
            # Complaint info
            'complaint_data': complaint_info,
            'final_lead_time_days': final_reply.get('lead_time_days') if final_reply else None,
            
            # Raw data
            '_raw_qa_pairs': [qa_pair],
            '_raw_reply_analysis': reply_analysis
        }
        
        # Add recommendation
        if not reply_analysis['requirement_compliant']:
            result['recommendation'] = 'Missing required replies'
        elif qa_pair.get('customer_leave'):
            result['recommendation'] = 'Customer left conversation'
        else:
            result['recommendation'] = 'Meets requirements'
        
        return result

    def _create_ticket_result(self, ticket_id, status, reason, extra_data):
        """Create result object untuk failed analysis"""
        result = {
            'ticket_id': ticket_id,
            'status': status,
            'failure_reason': reason if status == 'failed' else None,
            'analysis_timestamp': datetime.now()
        }
        result.update(extra_data)
        return result

    def analyze_all_tickets(self, df, max_tickets=None):
        """Analisis semua tickets"""
        print("🚀 STARTING ANALYSIS PIPELINE (NEW REQUIREMENTS)")
        
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
        self._print_summary_report()
        
        return self.results, self.analysis_stats

    def _calculate_stats(self, total_tickets):
        """Hitung statistics"""
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
            
            # Performance distribution
            performances = [r['performance_rating'] for r in successful]
            stats['performance_distribution'] = dict(Counter(performances))
            
            # Lead time statistics
            first_lead_times = [r['first_reply_lead_time_minutes'] for r in successful if r.get('first_reply_lead_time_minutes')]
            final_lead_times = [r['final_reply_lead_time_minutes'] for r in successful if r.get('final_reply_lead_time_minutes')]
            
            stats['overall_lead_times'] = {
                'first_reply_avg_minutes': np.mean(first_lead_times) if first_lead_times else 0,
                'final_reply_avg_minutes': np.mean(final_lead_times) if final_lead_times else 0,
                'first_reply_samples': len(first_lead_times),
                'final_reply_samples': len(final_lead_times)
            }
            
            # Reply effectiveness
            stats['reply_effectiveness'] = {
                'first_reply_found_rate': sum(1 for r in successful if r['first_reply_found']) / len(successful),
                'final_reply_found_rate': sum(1 for r in successful if r['final_reply_found']) / len(successful),
                'customer_leave_cases': sum(1 for r in successful if r.get('customer_leave', False))
            }
        
        return stats

    def _print_summary_report(self):
        """Print summary report"""
        stats = self.analysis_stats
        
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print(f"📊 Total Tickets: {stats['total_tickets']}")
        print(f"✅ Successful: {stats['successful_analysis']} ({stats['success_rate']*100:.1f}%)")
        
        if 'issue_type_distribution' in stats:
            print(f"🎯 Issue Types: {stats['issue_type_distribution']}")
        
        if 'overall_lead_times' in stats:
            lt = stats['overall_lead_times']
            print(f"⏱️ Lead Times - First: {lt['first_reply_avg_minutes']:.1f}min, Final: {lt['final_reply_avg_minutes']:.1f}min")

# Initialize Pipeline
pipeline = CompleteAnalysisPipeline()

print("✅ NEW REQUIREMENTS Analysis Pipeline Ready!")
print("   ✓ Simplified logic: 1 main question per ticket")
print("   ✓ Role support: Customer, Operator, Ticket Automation, Blank")
print("   ✓ Complaint matching dengan phone number")
print("   ✓ Customer leave detection dengan timeout 3 menit")
print("=" * 60)
