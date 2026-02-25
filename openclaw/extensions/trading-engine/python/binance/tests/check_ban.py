from datetime import datetime, timezone, timedelta
KST = timezone(timedelta(hours=9))
now = datetime.now(KST)
ban_ts = 1771975758291 / 1000
ban_dt = datetime.fromtimestamp(ban_ts, tz=KST)
print(f"Now (KST): {now}")
print(f"Ban expires (KST): {ban_dt}")
wait = (ban_dt - now).total_seconds()
print(f"Wait: {wait:.0f}s ({wait/60:.1f}min)")
