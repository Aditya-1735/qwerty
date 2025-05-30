import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import time
from datetime import datetime
import hashlib

# Configuration
# Replace the hardcoded key with:
GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]  # Your production API key
DEFAULT_RADIUS = 10000  # 10km default radius
MAX_RESULTS = 20

# Page configuration
st.set_page_config(
    page_title="MedLocator - Professional Hospital Finder",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --background-color: #FFFFFF;
        --secondary-background-color: #F0F2F6;
        --text-color: #262730;
    }
    .main-header {
        text-align: center;
        padding:1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .hospital-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .status-open { color: #28a745; font-weight: bold; }
    .status-closed { color: #dc3545; font-weight: bold; }
    .status-unknown { color: #6c757d; font-weight: bold; }
    .rating-excellent { color: #28a745; }
    .rating-good { color: #ffc107; }
    .rating-poor { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 MedLocator</h1>
    <p>Professional Hospital & Medical Facility Finder</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Search Configuration")
    
    # Search settings
    st.subheader("🎯 Search Parameters")
    radius = st.select_slider(
        "Search Radius",
        options=[1000, 2500, 5000, 10000, 15000, 25000, 50000],
        value=DEFAULT_RADIUS,
        format_func=lambda x: f"{x/1000:.1f} km"
    )
    
    max_results = st.selectbox(
        "Maximum Results",
        options=[5, 10, 15, 20],
        index=3
    )
    
    # Facility types
    st.subheader("🏥 Facility Types")
    include_hospitals = st.checkbox("Hospitals", value=True)
    include_clinics = st.checkbox("Medical Clinics", value=False)
    include_urgent_care = st.checkbox("Urgent Care", value=False)
    
    # Filters
    st.subheader("🔍 Filters")
    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)
    only_open = st.checkbox("Only show currently open facilities")
    
    # About section
    with st.expander("ℹ️ About MedLocator"):
        st.markdown("""
        **MedLocator** is a professional tool for finding nearby medical facilities.
        
        **Features:**
        - Real-time hospital data
        - Interactive mapping
        - Detailed facility information
        - Export capabilities
        - Mobile-responsive design
        
        **Data Source:** Places API (New)
        """)

def geocode_address(address):
    """Convert address to coordinates using Geocoding API"""
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            formatted_address = data['results'][0]['formatted_address']
            return location['lat'], location['lng'], formatted_address
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

def search_medical_facilities(lat, lng, radius, max_results, facility_types):
    """Search for nearby medical facilities using Places API (New)"""
    try:
        url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.businessStatus,places.regularOpeningHours,places.nationalPhoneNumber,places.websiteUri,places.googleMapsUri,places.types,places.id,places.photos'
        }
        
        data = {
            "includedTypes": facility_types,
            "maxResultCount": max_results,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": radius
                }
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('places', [])
        else:
            st.error(f"API Error: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Error searching facilities: {e}")
        return []

def process_facility_data(places, min_rating=1.0, only_open=False):
    """Process and filter facility data"""
    facilities = []
    
    for place in places:
        # Extract basic info
        facility_info = {
            'name': place.get('displayName', {}).get('text', 'Unknown'),
            'address': place.get('formattedAddress', 'Address not available'),
            'lat': place.get('location', {}).get('latitude', 0),
            'lng': place.get('location', {}).get('longitude', 0),
            'rating': place.get('rating', 0),
            'user_ratings_total': place.get('userRatingCount', 0),
            'business_status': place.get('businessStatus', 'Unknown'),
            'place_id': place.get('id', ''),
            'types': place.get('types', []),
            'phone': place.get('nationalPhoneNumber', 'Not available'),
            'website': place.get('websiteUri', 'Not available'),
            'google_maps_url': place.get('googleMapsUri', 'Not available')
        }
        
        # Extract opening hours
        opening_hours = place.get('regularOpeningHours', {})
        if opening_hours:
            facility_info['is_open'] = opening_hours.get('openNow', None)
            weekday_descriptions = opening_hours.get('weekdayDescriptions', [])
            facility_info['hours'] = '\n'.join(weekday_descriptions) if weekday_descriptions else 'Hours not available'
        else:
            facility_info['is_open'] = None
            facility_info['hours'] = 'Hours not available'
        
        # Apply filters
        if facility_info['rating'] > 0 and facility_info['rating'] < min_rating:
            continue
            
        if only_open and facility_info['is_open'] != True:
            continue
        
        facilities.append(facility_info)
    
    return facilities

def create_professional_map(center_lat, center_lng, facilities):
    """Create an enhanced folium map with professional styling"""
    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=13,
        tiles='CartoDB positron'
    )
    
    # Add search center marker
    folium.Marker(
        [center_lat, center_lng],
        popup=folium.Popup("📍 Search Center", max_width=200),
        icon=folium.Icon(color='red', icon='star', prefix='fa')
    ).add_to(m)
    
    # Add facility markers
    for facility in facilities:
        # Determine marker color and icon based on type and rating
        facility_types = facility['types']
        
        if 'hospital' in facility_types:
            icon = 'plus-square'
            base_color = 'blue'
        elif 'clinic' in facility_types or 'doctor' in facility_types:
            icon = 'stethoscope'
            base_color = 'green'
        else:
            icon = 'medkit'
            base_color = 'orange'
        
        # Adjust color based on rating
        if facility['rating'] >= 4.5:
            color = 'darkgreen'
        elif facility['rating'] >= 4.0:
            color = base_color
        elif facility['rating'] >= 3.0:
            color = 'orange'
        elif facility['rating'] > 0:
            color = 'red'
        else:
            color = 'gray'
        
        # Create comprehensive popup
        popup_html = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{facility['name']}</h4>
            <p style="margin: 5px 0;"><b>📍 Address:</b><br>{facility['address']}</p>
        """
        
        if facility['rating'] > 0:
            stars = '⭐' * int(facility['rating'])
            popup_html += f'<p style="margin: 5px 0;"><b>Rating:</b> {facility["rating"]} {stars} ({facility["user_ratings_total"]} reviews)</p>'
        
        if facility['phone'] != 'Not available':
            popup_html += f'<p style="margin: 5px 0;"><b>📞 Phone:</b> {facility["phone"]}</p>'
        
        # Status indicator
        if facility['is_open'] == True:
            popup_html += '<p style="margin: 5px 0; color: green;"><b>Status:</b> 🟢 Open Now</p>'
        elif facility['is_open'] == False:
            popup_html += '<p style="margin: 5px 0; color: red;"><b>Status:</b> 🔴 Closed</p>'
        
        if facility['website'] != 'Not available':
            popup_html += f'<p style="margin: 5px 0;"><a href="{facility["website"]}" target="_blank" style="color: #007bff;">🌐 Website</a></p>'
        
        popup_html += f'<p style="margin: 5px 0;"><a href="{facility["google_maps_url"]}" target="_blank" style="color: #007bff;">🗺️ Directions</a></p>'
        popup_html += "</div>"
        
        folium.Marker(
            [facility['lat'], facility['lng']],
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    return m

def get_facility_types():
    """Get selected facility types based on checkboxes"""
    types = []
    if include_hospitals:
        types.append("hospital")
    if include_clinics:
        types.extend(["clinic", "doctor"])
    if include_urgent_care:
        types.append("urgent_care")
    
    if not types:  # Default to hospitals if nothing selected
        types = ["hospital"]
    
    return types

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Location Search")
    
    # Enhanced location input
    input_method = st.radio(
        "Search Method:",
        ["🏠 Address/City", "🎯 Coordinates", "📱 Current Location*"],
        horizontal=True
    )
    
    if input_method == "🏠 Address/City":
        location_input = st.text_input(
            "Enter location:",
            placeholder="e.g., Times Square, New York or 123 Main St, Boston, MA",
            help="Enter any address, landmark, or city name"
        )
    elif input_method == "🎯 Coordinates":
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            latitude = st.number_input("Latitude", format="%.6f", value=40.7128)
        with coord_col2:
            longitude = st.number_input("Longitude", format="%.6f", value=-74.0060)
    else:
        st.info("📱 Current location feature requires GPS access. Use coordinates as alternative.")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            latitude = st.number_input("Latitude", format="%.6f", value=40.7128)
        with coord_col2:
            longitude = st.number_input("Longitude", format="%.6f", value=-74.0060)

with col2:
    st.subheader("🚀 Quick Actions")
    
    # Search button with enhanced styling
    search_button = st.button(
        "🔍 Find Medical Facilities", 
        type="primary",
        use_container_width=True
    )

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_location' not in st.session_state:
    st.session_state.search_location = None

# Main search logic
if search_button:
    try:
        # Get coordinates
        if input_method == "🏠 Address/City":
            if not location_input:
                st.error("⚠️ Please enter a location to search")
                st.stop()
            
            with st.spinner("🌍 Locating address..."):
                lat, lng, formatted_address = geocode_address(location_input)
            
            if lat is None or lng is None:
                st.error("❌ Could not find the specified location. Please try a different address.")
                st.stop()
        else:
            lat, lng = latitude, longitude
            formatted_address = f"Coordinates: {lat:.6f}, {lng:.6f}"
        
        # Get facility types
        facility_types = get_facility_types()
        
        # Search for medical facilities
        with st.spinner("🔍 Searching for medical facilities..."):
            places = search_medical_facilities(lat, lng, radius, max_results, facility_types)
        
        if not places:
            st.warning("⚠️ No medical facilities found in the specified area. Try adjusting your search parameters.")
            st.session_state.search_results = None
        else:
            # Process and filter data
            facilities = process_facility_data(places, min_rating, only_open)
            
            if not facilities:
                st.warning("⚠️ No facilities match your current filters. Try adjusting the minimum rating or 'open only' filter.")
                st.session_state.search_results = None
            else:
                # Store results in session state
                st.session_state.search_results = {
                    'facilities': facilities,
                    'location': {'lat': lat, 'lng': lng, 'address': formatted_address},
                    'search_params': {
                        'radius': radius,
                        'max_results': max_results,
                        'facility_types': facility_types,
                        'min_rating': min_rating,
                        'only_open': only_open
                    }
                }
                
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        st.info("💡 Please try again or contact support if the issue persists.")
        st.session_state.search_results = None

# Display results if they exist in session state
if st.session_state.search_results:
    results = st.session_state.search_results
    facilities = results['facilities']
    location = results['location']
    
    # Success message
    st.success(f"📍 **Search Location:** {location['address']}")
    st.success(f"✅ Found **{len(facilities)}** medical facilities!")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "📊 Analytics"])
    
    with tab1:
        st.subheader("Interactive Map")
        facility_map = create_professional_map(location['lat'], location['lng'], facilities)
        st_folium(facility_map, width=None, height=600)
        
        # Enhanced legend
        st.markdown("""
        **🗺️ Map Legend:**
        - 🔴 **Red Star:** Your search location
        - 🟢 **Dark Green:** Excellent rating (4.5+)
        - 🔵 **Blue/Green:** Good rating (4.0+)
        - 🟠 **Orange:** Average rating (3.0+)
        - 🔴 **Red:** Below average rating
        - ⚫ **Gray:** No rating available
        
        **Icons:** 🏥 Hospital | 🩺 Clinic | 🚑 Urgent Care
        """)
    
    with tab2:
        st.subheader("Detailed Facility List")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Rating (High to Low)", "Rating (Low to High)", "Name (A-Z)", "Reviews Count"]
        )
        
        # Sort facilities
        facilities_sorted = facilities.copy()
        if sort_by == "Rating (High to Low)":
            facilities_sorted.sort(key=lambda x: x['rating'], reverse=True)
        elif sort_by == "Rating (Low to High)":
            facilities_sorted.sort(key=lambda x: x['rating'])
        elif sort_by == "Name (A-Z)":
            facilities_sorted.sort(key=lambda x: x['name'])
        elif sort_by == "Reviews Count":
            facilities_sorted.sort(key=lambda x: x['user_ratings_total'], reverse=True)
        
        # Display facilities
        for i, facility in enumerate(facilities_sorted, 1):
            # Status indicator
            if facility['is_open'] == True:
                status = "🟢 Open"
                status_class = "status-open"
            elif facility['is_open'] == False:
                status = "🔴 Closed"
                status_class = "status-closed"
            else:
                status = "⚫ Hours Unknown"
                status_class = "status-unknown"
            
            # Rating display
            if facility['rating'] > 0:
                if facility['rating'] >= 4.0:
                    rating_class = "rating-excellent"
                elif facility['rating'] >= 3.0:
                    rating_class = "rating-good"
                else:
                    rating_class = "rating-poor"
                rating_display = f"{facility['rating']} ⭐ ({facility['user_ratings_total']} reviews)"
            else:
                rating_class = ""
                rating_display = "No rating"
            
            with st.expander(f"{i}. {facility['name']} - {status}"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.write(f"**📍 Address:** {facility['address']}")
                    st.write(f"**⭐ Rating:** {rating_display}")
                    st.write(f"**📞 Phone:** {facility['phone']}")
                    st.write(f"**🏥 Type:** {', '.join(facility['types'])}")
                
                with col_b:
                    if facility['website'] != 'Not available':
                        st.link_button("🌐 Website", facility['website'])
                    
                    st.link_button("🗺️ Directions", facility['google_maps_url'])
                
                if facility['hours'] != 'Hours not available':
                    st.markdown("**🕒 Opening Hours:**")
                    st.text(facility['hours'])
    
    with tab3:
        st.subheader("Search Analytics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏥 Total Facilities", len(facilities))
        
        with col2:
            rated_facilities = [f for f in facilities if f['rating'] > 0]
            if rated_facilities:
                avg_rating = sum(f['rating'] for f in rated_facilities) / len(rated_facilities)
                st.metric("📊 Average Rating", f"{avg_rating:.1f} ⭐")
            else:
                st.metric("📊 Average Rating", "N/A")
        
        with col3:
            open_facilities = [f for f in facilities if f['is_open'] == True]
            st.metric("🟢 Currently Open", len(open_facilities))
        
        with col4:
            excellent_facilities = [f for f in facilities if f['rating'] >= 4.0]
            st.metric("⭐ Highly Rated (4.0+)", len(excellent_facilities))
        
        # Charts and additional analytics
        if rated_facilities:
            st.subheader("📈 Rating Distribution")
            
            # Create rating distribution
            rating_ranges = {
                "⭐⭐⭐⭐⭐ (4.5-5.0)": len([f for f in rated_facilities if f['rating'] >= 4.5]),
                "⭐⭐⭐⭐ (4.0-4.4)": len([f for f in rated_facilities if 4.0 <= f['rating'] < 4.5]),
                "⭐⭐⭐ (3.0-3.9)": len([f for f in rated_facilities if 3.0 <= f['rating'] < 4.0]),
                "⭐⭐ (2.0-2.9)": len([f for f in rated_facilities if 2.0 <= f['rating'] < 3.0]),
                "⭐ (1.0-1.9)": len([f for f in rated_facilities if 1.0 <= f['rating'] < 2.0])
            }
            
            df_ratings = pd.DataFrame(list(rating_ranges.items()), columns=['Rating Range', 'Count'])
            st.bar_chart(df_ratings.set_index('Rating Range'))
    
    # Export functionality
    st.subheader("📥 Export Data")
    
    # Prepare data for export
    export_data = []
    for facility in facilities:
        export_data.append({
            'Name': facility['name'],
            'Address': facility['address'],
            'Phone': facility['phone'],
            'Rating': facility['rating'] if facility['rating'] > 0 else 'Not rated',
            'Total Reviews': facility['user_ratings_total'],
            'Currently Open': facility['is_open'],
            'Business Status': facility['business_status'],
            'Website': facility['website'],
            'Google Maps URL': facility['google_maps_url'],
            'Types': ', '.join(facility['types']),
            'Latitude': facility['lat'],
            'Longitude': facility['lng']
        })
    
    df_export = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df_export.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"medical_facilities_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = df_export.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"medical_facilities_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Performance metrics
    with st.expander("⚡ Performance Metrics"):
        st.success(f"✅ Search completed successfully")
        st.info(f"🔍 Searched within {results['search_params']['radius']/1000}km radius")
        st.info(f"✨ Found {len(facilities)} matching facilities")
        
    # Clear results button
    if st.button("🗑️ Clear Results", type="secondary"):
        st.session_state.search_results = None
        st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏥 MedLocator**")
    st.caption("Professional Medical Facility Finder")

with col2:
    st.markdown("**⚡ Powered by:**")
    st.caption("Places API (New)")

with col3:
    st.markdown("**📞 Support:**")
    st.caption("For technical issues or feedback")

# Help section
with st.expander("❓ Help & FAQ"):
    st.markdown("""
    ### Frequently Asked Questions
    
    **Q: How accurate is the data?**
    A: Data is sourced directly from Google Places API and updated in real-time.
    
    **Q: Can I search for specific types of medical facilities?**
    A: Yes! Use the sidebar to select hospitals, clinics, or urgent care centers.
    
    **Q: What do the map colors mean?**
    A: Colors indicate rating quality - green for excellent (4.5+), blue for good (4.0+), orange for average (3.0+), red for below average, and gray for unrated.
    
    **Q: Can I save my search results?**
    A: Yes! Use the export function to download results as CSV or JSON format.
    
    **Q: Is my location data stored?**
    A: No, all searches are processed in real-time and no personal data is stored.
    
    ### Tips for Better Results
    - Use specific addresses for more precise results
    - Adjust the search radius based on your needs
    - Try different facility type combinations
    - Use filters to narrow down results
    """)