"""
Helper script to push the clean committed code to GitHub.

Usage:
    python push_to_github.py
    or
    python push_to_github.py --token YOUR_GITHUB_PERSONAL_ACCESS_TOKEN
"""
import sys
import getpass
import argparse
from pathlib import Path
from dulwich.repo import Repo
import dulwich.porcelain as porcelain

REPO_ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Push code to GitHub repository.")
    parser.add_argument("--token", help="GitHub Personal Access Token (PAT)")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("Pushing to https://github.com/MeGH2711/isles26_stroke_lesion_segmentation")
        print("Please enter your GitHub Personal Access Token (PAT with 'repo' scope):")
        try:
            token = input("GitHub PAT: ").strip()
        except Exception:
            token = getpass.getpass("GitHub PAT: ").strip()

    if not token:
        print("No token provided. Aborted.")
        sys.exit(1)

    auth_url = f"https://MeGH2711:{token}@github.com/MeGH2711/isles26_stroke_lesion_segmentation.git"
    repo = Repo(str(REPO_ROOT))

    print("Pushing 'main' branch to GitHub...")
    try:
        porcelain.push(repo, auth_url, "refs/heads/main:refs/heads/main")
        print("\n[SUCCESS] Successfully pushed code to https://github.com/MeGH2711/isles26_stroke_lesion_segmentation!")
    except Exception as e:
        print(f"\n[ERROR] Push failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
