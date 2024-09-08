# Databricks notebook source
import streamlit as st
import hashlib

# Define users and passwords
USERS = {
    "Alain": hashlib.sha256("ibc_2024".encode()).hexdigest(),
}

def verify_password(password, hashed_password):
    return hashlib.sha256(password.encode()).hexdigest() == hashed_password

def main():
    st.title("Test App")
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USERS and verify_password(password, USERS[username]):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
        st.write("You are logged in!")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    main()