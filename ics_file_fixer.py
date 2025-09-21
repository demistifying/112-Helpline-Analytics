# ics_file_fixer.py - Script to fix or replace corrupted ICS file
import os
import shutil
from datetime import datetime

def backup_corrupted_file():
    """Backup the corrupted ICS file"""
    original_path = os.path.join("data", "festivals.ics")
    
    if os.path.exists(original_path):
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join("data", f"festivals_backup_{timestamp}.ics")
        
        try:
            shutil.copy2(original_path, backup_path)
            print(f"✅ Backed up corrupted file to: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"❌ Error creating backup: {str(e)}")
            return None
    else:
        print("ℹ️ Original festivals.ics file not found")
        return None

def create_clean_festivals_ics():
    """Create a clean festivals.ics file with major Indian festivals"""
    
    # Clean ICS content with major Indian festivals for 2025
    clean_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Goa Police//Festival Calendar//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
X-WR-CALNAME:Indian Festivals 2025
X-WR-TIMEZONE:Asia/Kolkata
X-WR-CALDESC:Major Indian Festivals and Public Holidays

BEGIN:VEVENT
UID:republic-day-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250126
DTEND;VALUE=DATE:20250127
SUMMARY:Republic Day
DESCRIPTION:Republic Day of India
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:maha-shivratri-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250226
DTEND;VALUE=DATE:20250227
SUMMARY:Maha Shivratri
DESCRIPTION:Hindu festival dedicated to Lord Shiva
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:holi-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250313
DTEND;VALUE=DATE:20250314
SUMMARY:Holi
DESCRIPTION:Festival of Colors
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:good-friday-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250418
DTEND;VALUE=DATE:20250419
SUMMARY:Good Friday
DESCRIPTION:Christian festival commemorating crucifixion of Jesus Christ
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:ram-navami-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250406
DTEND;VALUE=DATE:20250407
SUMMARY:Ram Navami
DESCRIPTION:Hindu festival celebrating birth of Lord Rama
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:eid-ul-fitr-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250331
DTEND;VALUE=DATE:20250401
SUMMARY:Eid ul-Fitr
DESCRIPTION:Islamic festival marking end of Ramadan
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:independence-day-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250815
DTEND;VALUE=DATE:20250816
SUMMARY:Independence Day
DESCRIPTION:Independence Day of India
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:janmashtami-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250816
DTEND;VALUE=DATE:20250817
SUMMARY:Janmashtami
DESCRIPTION:Hindu festival celebrating birth of Lord Krishna
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:ganesh-chaturthi-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20250829
DTEND;VALUE=DATE:20250830
SUMMARY:Ganesh Chaturthi
DESCRIPTION:Hindu festival dedicated to Lord Ganesha
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:dussehra-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251002
DTEND;VALUE=DATE:20251003
SUMMARY:Dussehra
DESCRIPTION:Hindu festival celebrating victory of good over evil
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:gandhi-jayanti-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251002
DTEND;VALUE=DATE:20251003
SUMMARY:Gandhi Jayanti
DESCRIPTION:Birthday of Mahatma Gandhi
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:diwali-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251020
DTEND;VALUE=DATE:20251021
SUMMARY:Diwali
DESCRIPTION:Festival of Lights
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:bhai-dooj-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251022
DTEND;VALUE=DATE:20251023
SUMMARY:Bhai Dooj
DESCRIPTION:Hindu festival celebrating bond between brothers and sisters
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:guru-nanak-jayanti-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251115
DTEND;VALUE=DATE:20251116
SUMMARY:Guru Nanak Jayanti
DESCRIPTION:Birthday of Guru Nanak Dev Ji
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

BEGIN:VEVENT
UID:christmas-2025@goapolice.gov.in
DTSTART;VALUE=DATE:20251225
DTEND;VALUE=DATE:20251226
SUMMARY:Christmas
DESCRIPTION:Christian festival celebrating birth of Jesus Christ
LOCATION:India
STATUS:CONFIRMED
TRANSP:OPAQUE
END:VEVENT

END:VCALENDAR"""

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Write the clean file
    clean_path = os.path.join("data", "festivals.ics")
    
    try:
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        print(f"✅ Created clean festivals.ics file at: {clean_path}")
        return clean_path
    except Exception as e:
        print(f"❌ Error creating clean ICS file: {str(e)}")
        return None

def fix_festivals_file():
    """Main function to fix the festivals file"""
    print("🔧 Starting ICS file fix process...")
    
    # Step 1: Backup corrupted file
    backup_path = backup_corrupted_file()
    
    # Step 2: Create clean file
    clean_path = create_clean_festivals_ics()
    
    if clean_path:
        print("✅ Festival file fix completed successfully!")
        print("You can now run your dashboard without encoding errors.")
        return True
    else:
        print("❌ Failed to fix festival file")
        return False

if __name__ == "__main__":
    fix_festivals_file()