"""Print the items of the released VIKI-R Hugging Face collection (checkpoints), so you can choose which to download."""
from huggingface_hub import get_collection

SLUG = "henggg/viki-r-68ef92c3a67e90535ecc1a9b"       # from the upstream README
c = get_collection(SLUG)
print(f"collection: {c.title}  ({len(c.items)} items)")
for it in c.items:
    print(f"  {it.item_type:8s} {it.item_id}")
