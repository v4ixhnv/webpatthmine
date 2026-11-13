"""
Generates high-fidelity sample datasets for WEBPATHMINE:
1. online_shoppers_intention.csv (UCI Online Shoppers schema)
2. clickstream_sample.csv (timestamped session clickstream logs)
"""
import os
import numpy as np
import pandas as pd

def generate_online_shoppers_data(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # Revenue flag: ~16% positive rate (realistic e-commerce conversion)
    revenue = np.random.choice([False, True], size=n_samples, p=[0.84, 0.16])
    
    # Page values generated from browsing engagement, completely INDEPENDENT of revenue
    # (P0 FIX: Eliminates artificial target leakage from synthetic generator)
    page_values = np.zeros(n_samples)
    for i in range(n_samples):
        # Realistic zero-inflated distribution independent of future conversion outcome
        if np.random.rand() < 0.70:
            page_values[i] = 0.0
        else:
            page_values[i] = np.random.exponential(scale=6.0)
    page_values = np.round(page_values, 2)
    
    # Exit and Bounce rates (negatively correlate with conversion)
    bounce_rates = np.zeros(n_samples)
    exit_rates = np.zeros(n_samples)
    for i in range(n_samples):
        if revenue[i]:
            bounce_rates[i] = np.clip(np.random.beta(1.5, 30.0), 0.0, 0.2)
            exit_rates[i] = np.clip(bounce_rates[i] + np.random.beta(2.0, 20.0), 0.01, 0.2)
        else:
            bounce_rates[i] = np.clip(np.random.beta(2.0, 15.0), 0.0, 0.2)
            exit_rates[i] = np.clip(bounce_rates[i] + np.random.beta(2.5, 10.0), 0.02, 0.2)
    bounce_rates = np.round(bounce_rates, 4)
    exit_rates = np.round(exit_rates, 4)
    
    # Browsing counts and durations
    admin_counts = np.random.negative_binomial(1, 0.3, size=n_samples)
    admin_duration = np.round(admin_counts * np.random.gamma(2.0, 25.0, size=n_samples), 2)
    
    info_counts = np.random.negative_binomial(1, 0.6, size=n_samples)
    info_duration = np.round(info_counts * np.random.gamma(1.5, 20.0, size=n_samples), 2)
    
    # Product related is higher for converters
    prod_counts = np.zeros(n_samples, dtype=int)
    prod_duration = np.zeros(n_samples)
    for i in range(n_samples):
        if revenue[i]:
            prod_counts[i] = int(np.random.gamma(5.0, 10.0)) + 5
            prod_duration[i] = np.round(prod_counts[i] * np.random.gamma(3.0, 18.0), 2)
        else:
            prod_counts[i] = int(np.random.gamma(2.5, 8.0)) + 1
            prod_duration[i] = np.round(prod_counts[i] * np.random.gamma(2.0, 15.0), 2)
            
    months = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_weights = [0.02, 0.15, 0.25, 0.04, 0.04, 0.04, 0.05, 0.06, 0.25, 0.10]
    month = np.random.choice(months, size=n_samples, p=month_weights)
    
    special_day = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n_samples, p=[0.88, 0.03, 0.03, 0.02, 0.02, 0.02])
    operating_systems = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_samples, p=[0.21, 0.54, 0.20, 0.02, 0.01, 0.01, 0.005, 0.005])
    browser = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_samples, p=[0.20, 0.65, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005])
    region = np.random.choice(list(range(1, 10)), size=n_samples)
    traffic_type = np.random.choice(list(range(1, 21)), size=n_samples, p=[0.20, 0.32, 0.16, 0.09] + [0.23/16]*16)
    visitor_type = np.random.choice(['Returning_Visitor', 'New_Visitor', 'Other'], size=n_samples, p=[0.85, 0.14, 0.01])
    weekend = np.random.choice([False, True], size=n_samples, p=[0.77, 0.23])
    
    df = pd.DataFrame({
        'Administrative': admin_counts,
        'Administrative_Duration': admin_duration,
        'Informational': info_counts,
        'Informational_Duration': info_duration,
        'ProductRelated': prod_counts,
        'ProductRelated_Duration': prod_duration,
        'BounceRates': bounce_rates,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_day,
        'Month': month,
        'OperatingSystems': operating_systems,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type,
        'Weekend': weekend,
        'Revenue': revenue
    })
    return df

def generate_clickstream_data(n_sessions=800, random_state=42):
    np.random.seed(random_state)
    
    pages = [
        ('home', 'Home'),
        ('search', 'Search'),
        ('catalog_electronics', 'Catalog'),
        ('catalog_apparel', 'Catalog'),
        ('product_detail', 'Product'),
        ('reviews', 'Product'),
        ('cart', 'Cart'),
        ('checkout_shipping', 'Checkout'),
        ('checkout_payment', 'Checkout'),
        ('order_confirmation', 'Checkout'),
        ('account_profile', 'Account'),
        ('help_faq', 'Support'),
        ('promo_landing', 'Promotion')
    ]
    
    # Typical navigation patterns
    patterns = [
        # Buyer journey
        ['home', 'catalog_electronics', 'product_detail', 'reviews', 'cart', 'checkout_shipping', 'checkout_payment', 'order_confirmation'],
        ['search', 'product_detail', 'cart', 'checkout_shipping', 'checkout_payment', 'order_confirmation'],
        ['promo_landing', 'catalog_apparel', 'product_detail', 'cart', 'checkout_shipping', 'checkout_payment', 'order_confirmation'],
        # Window shopper / Abandoner
        ['home', 'search', 'catalog_electronics', 'product_detail', 'cart'],
        ['home', 'catalog_apparel', 'product_detail', 'reviews', 'cart'],
        ['promo_landing', 'product_detail', 'cart'],
        # Researcher
        ['home', 'search', 'product_detail', 'reviews', 'help_faq'],
        ['catalog_electronics', 'product_detail', 'reviews', 'product_detail'],
        # Direct bounce
        ['home'],
        ['search', 'home'],
        ['promo_landing'],
        ['account_profile', 'home']
    ]
    pattern_weights = [0.08, 0.05, 0.05, 0.14, 0.12, 0.08, 0.12, 0.10, 0.10, 0.08, 0.05, 0.03]
    
    records = []
    base_time = pd.Timestamp('2024-03-01 08:00:00')
    
    for sess_idx in range(1, n_sessions + 1):
        sess_id = f"SESS_{sess_idx:05d}"
        chosen_pattern_idx = np.random.choice(len(patterns), p=pattern_weights)
        path = patterns[chosen_pattern_idx].copy()
        
        # Add random variations
        if len(path) > 1 and np.random.rand() < 0.25:
            # Maybe revisits reviews or search
            insert_page = np.random.choice(['search', 'reviews', 'catalog_electronics'])
            insert_pos = np.random.randint(1, len(path))
            path.insert(insert_pos, insert_page)
            
        sess_start = base_time + pd.Timedelta(minutes=np.random.randint(0, 1440 * 14), seconds=np.random.randint(0, 59))
        curr_time = sess_start
        
        for step, page in enumerate(path):
            dwell = int(np.random.exponential(scale=45.0) + 10)
            category = next((cat for p, cat in pages if p == page), 'Other')
            
            event_type = 'page_view'
            if page == 'cart':
                event_type = 'add_to_cart'
            elif page == 'order_confirmation':
                event_type = 'purchase'
            elif step == len(path) - 1:
                event_type = 'exit'
                
            records.append({
                'session_id': sess_id,
                'timestamp': curr_time.strftime('%Y-%m-%d %H:%M:%S'),
                'page_path': f"/{page}",
                'page_name': page,
                'page_category': category,
                'event_type': event_type,
                'dwell_time_seconds': dwell
            })
            curr_time += pd.Timedelta(seconds=dwell + np.random.randint(2, 8))
            
    return pd.DataFrame(records)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("Generating UCI Online Shoppers Intention sample data...")
    df_shoppers = generate_online_shoppers_data(n_samples=3000, random_state=42)
    df_shoppers.to_csv('data/online_shoppers_intention.csv', index=False)
    print(f"Saved data/online_shoppers_intention.csv with shape {df_shoppers.shape}")
    
    print("Generating Clickstream sample data...")
    df_clicks = generate_clickstream_data(n_sessions=1000, random_state=42)
    df_clicks.to_csv('data/clickstream_sample.csv', index=False)
    print(f"Saved data/clickstream_sample.csv with shape {df_clicks.shape}")
    print("Sample generation complete!")
