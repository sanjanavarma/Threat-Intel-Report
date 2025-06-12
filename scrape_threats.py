import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from transformers import pipeline # Import the pipeline for summarization

# Initialize the summarization pipeline globally to avoid re-loading for each article
# Using 'sshleifer/distilbart-cnn-12-6' as a good general-purpose summarization model.
# This model needs to be downloaded the first time it's used.
try:
    summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("Hugging Face summarization pipeline initialized successfully.")
except Exception as e:
    summarizer_pipeline = None
    print(f"Failed to initialize Hugging Face summarization pipeline: {e}")
    print("Summaries will fall back to extractive method.")


def fetch_page_content(url):
    """
    Fetches the content of a given URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        BeautifulSoup object or None: Parsed HTML content if successful, otherwise None.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_articles_by_date(main_url, target_date_str="June 10, 2025"):
    """
    Extracts article links from the main CISA advisories page,
    filtering them by a specific publication date.

    Args:
        main_url (str): The URL of the main advisories page.
        target_date_str (str): The date string to filter articles by (e.g., "June 10, 2025").

    Returns:
        list: A list of dictionaries, each containing 'title' and 'url' of matching articles.
    """
    print(f"Searching for articles published on: {target_date_str}")
    target_date = datetime.strptime(target_date_str, "%B %d, %Y").date()
    soup = fetch_page_content(main_url)
    if not soup:
        return []

    articles_found = []
    # Adjusting selectors based on the provided HTML structure
    # Each article is within an <article> tag with class 'c-teaser'
    advisory_items = soup.find_all('article', class_='c-teaser')

    if not advisory_items:
        print("No advisory items found. Check the HTML structure and selectors.")
        return []

    for item in advisory_items:
        # Find the link (<a> tag) within the <h3> with class 'c-teaser__title'
        link_h3 = item.find('h3', class_='c-teaser__title')
        link_tag = link_h3.find('a', href=True) if link_h3 else None

        if not link_tag:
            continue

        article_url = link_tag['href']
        # CISA URLs might be relative, make them absolute
        if not article_url.startswith('http'):
            article_url = requests.compat.urljoin(main_url, article_url)

        article_title = link_tag.get_text(strip=True)

        # Find the date string within the <time> tag inside <div class="c-teaser__date">
        date_div = item.find('div', class_='c-teaser__date')
        time_tag = date_div.find('time') if date_div else None

        if time_tag:
            # Prioritize the 'datetime' attribute for robust parsing (ISO format)
            published_date_iso = time_tag.get('datetime')
            if published_date_iso:
                try:
                    # Parse ISO formatted datetime (e.g., 2025-06-10T12:00:00Z)
                    published_date = datetime.fromisoformat(published_date_iso.replace('Z', '+00:00')).date()
                except ValueError:
                    # Fallback to text content if isoformat fails (e.g., "Jun 10, 2025")
                    published_date_str = time_tag.get_text(strip=True)
                    try:
                        published_date = datetime.strptime(published_date_str, "%b %d, %Y").date()
                    except ValueError:
                        print(f"Could not parse date from text '{published_date_str}' for article: {article_title}")
                        continue
            else:
                # If no datetime attribute, use the text content
                published_date_str = time_tag.get_text(strip=True)
                try:
                    published_date = datetime.strptime(published_date_str, "%b %d, %Y").date()
                except ValueError:
                    print(f"Could not parse date from text '{published_date_str}' for article: {article_title}")
                    continue

            if published_date == target_date:
                articles_found.append({
                    'title': article_title,
                    'url': article_url
                })
        else:
            print(f"No date tag found for article: {article_title}")

    return articles_found

def get_article_summary(article_url):
    """
    Generates a summary from an individual article page using Hugging Face Transformers.

    Args:
        article_url (str): The URL of the article.

    Returns:
        str: A generated summary of the article, or a message indicating no summary.
    """
    soup = fetch_page_content(article_url)
    if not soup:
        return "No summary available (could not fetch content)."

    full_article_text = ""
    
    # Remove script and style tags to prevent their content from being extracted
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Define a list of possible content containers to try, ordered by specificity
    # Based on the provided screenshots and common website structures for article bodies.
    content_selectors = [
        'div.l-page-section_content', # Found in both screenshots for relevant content
        'div.l-page-section.l-page-section--rich-text', # A broader container, often holds content
        'div.layout__region--content', # Common for main content regions
        'div.field--name-body', # Drupal-specific main body field
        'div.node__content',    # General content node
        'article.c-article__body', # If CISA uses a specific article body class
        'main#main' # Broadest, only use if more specific fail, then refine within
    ]
    
    content_container = None
    for selector in content_selectors:
        found_container = soup.select_one(selector)
        if found_container:
            content_container = found_container
            break # Found a likely content container, stop searching

    # If a potential content container is found, extract text.
    if content_container:
        # Extract text from common textual elements within the identified content source.
        # Ensure spaces between elements for better readability and summarization input.
        # Iterate over descendants to get all relevant text.
        for element in content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'strong', 'em']):
            text_content = element.get_text(strip=True)
            
            # Skip if element is clearly part of boilerplate/navigation/metadata
            # Use stricter parent checks and more specific boilerplate patterns.
            is_irrelevant_parent = False
            # Check for common classes that indicate irrelevant sections
            irrelevant_parent_classes = [
                re.compile(r'c-site-footer|c-site-header|c-site-nav|view__filters|share-buttons|contact-info|related-links', re.IGNORECASE)
            ]
            for parent in element.parents:
                # Join the list of classes into a string for regex search
                parent_classes_str = ' '.join(parent.get('class', []))
                if any(cls.search(parent_classes_str) for cls in irrelevant_parent_classes):
                    is_irrelevant_parent = True
                    break
            
            # Also, explicitly check for boilerplate text patterns within the text content itself
            boilerplate_text_patterns = [
                r'report a cyber issue', r'secure by design', r'secure our world', 
                r'shields up', r'privacy policy', r'accessibility', r'sitemap',
                r'free cyber services', r'last updated', r'share this page',
                r'contact us', r'press release', r'disclaimer'
            ]

            if not is_irrelevant_parent and text_content and \
               not any(re.search(pattern, text_content, re.IGNORECASE) for pattern in boilerplate_text_patterns):
                full_article_text += text_content + " " # Add space to separate text from different elements
        full_article_text = full_article_text.strip() # Final strip after combining text

    # Check if the extracted text is meaningful for summarization
    min_summary_input_length = 100 # A reasonable minimum length for LLM input
    if not full_article_text or len(full_article_text) < min_summary_input_length:
        print(f"Warning: Extracted article text is too short ({len(full_article_text)} chars) or empty for {article_url}. Falling back to extractive summary.")
        # Fallback to extractive summary if input is too short or LLM fails
        # Use a more general content source for fallback to ensure something is returned
        fallback_content_source = content_container if content_container else soup 
        paragraphs = fallback_content_source.find_all('p')
        summary_parts = []
        char_count = 0
        max_chars = 500
        for p in paragraphs:
            paragraph_text = p.get_text(strip=True)
            if paragraph_text and char_count + len(paragraph_text) < max_chars:
                summary_parts.append(paragraph_text)
                char_count += len(paragraph_text)
            elif paragraph_text:
                remaining_chars = max_chars - char_count
                if remaining_chars > 0:
                    summary_parts.append(paragraph_text[:remaining_chars].rsplit(' ', 1)[0] + '...')
                break
        return " ".join(summary_parts) if summary_parts else "No summary available (extractive fallback)."

    # If LLM pipeline is initialized and text is long enough, use it
    if summarizer_pipeline:
        try:
            # Generate summary using the Hugging Face pipeline
            # max_length and min_length can be adjusted as needed
            # For very long texts, consider chunking or using models with larger context windows
            generated_summary = summarizer_pipeline(
                full_article_text, 
                max_length=150, 
                min_length=30, 
                do_sample=False
            )[0]['summary_text']
            return generated_summary
        except Exception as e:
            print(f"Error during LLM summarization for {article_url}: {e}")
            # Fallback to extractive summary if LLM summarization fails (e.g., CUDA issues, OOM)
            fallback_content_source = content_container if content_container else soup 
            paragraphs = fallback_content_source.find_all('p')
            summary_parts = []
            char_count = 0
            max_chars = 500
            for p in paragraphs:
                paragraph_text = p.get_text(strip=True)
                if paragraph_text and char_count + len(paragraph_text) < max_chars:
                    summary_parts.append(paragraph_text)
                    char_count += len(paragraph_text)
                elif paragraph_text:
                    remaining_chars = max_chars - char_count
                    if remaining_chars > 0:
                        summary_parts.append(paragraph_text[:remaining_chars].rsplit(' ', 1)[0] + '...')
                    break
            return " ".join(summary_parts) if summary_parts else "No summary available (extractive fallback)."
    else:
        # Fallback to extractive summary if LLM pipeline was not initialized at all
        fallback_content_source = content_container if content_container else soup 
        paragraphs = fallback_content_source.find_all('p')
        summary_parts = []
        char_count = 0
        max_chars = 500
        for p in paragraphs:
            paragraph_text = p.get_text(strip=True)
            if paragraph_text and char_count + len(paragraph_text) < max_chars:
                summary_parts.append(paragraph_text)
                char_count += len(paragraph_text)
            elif paragraph_text:
                remaining_chars = max_chars - char_count
                if remaining_chars > 0:
                    summary_parts.append(paragraph_text[:remaining_chars].rsplit(' ', 1)[0] + '...')
                break
        return " ".join(summary_parts) if summary_parts else "No summary available (extractive fallback)."


def main():
    """
    Main function to orchestrate the scraping process.
    """
    main_cisa_url = "https://www.cisa.gov/news-events/cybersecurity-advisories"
    target_date = "June 10, 2025" # Change this date as needed

    print(f"Starting CISA Advisory Scraper for {target_date}...")
    articles = extract_articles_by_date(main_cisa_url, target_date)

    if articles:
        print(f"\nFound {len(articles)} articles published on {target_date}:")
        for article in articles:
            print(f"\nTitle: {article['title']}")
            print(f"URL: {article['url']}")
            # Get and print the summary for each article
            summary = get_article_summary(article['url'])
            print(f"Summary: {summary}")
            print("-" * 50)
    else:
        print(f"No articles found for {target_date}.")

if __name__ == "__main__":
    main()
