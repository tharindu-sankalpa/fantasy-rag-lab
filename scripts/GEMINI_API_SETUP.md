# Gemini API Setup & Management Guide

This guide provides a comprehensive walkthrough for setting up access to the Google Gemini API (specifically Gemini 3) using simple command-line tools (`gcloud`). It covers authentication, project management, understanding configurations, and managing API keys.

---

## 1. Authentication

The first step is to authenticate your local machine with Google Cloud.

### Login via Browser

Run the following command to open a browser window where you can log in with your Google Account:

```bash
gcloud auth login
```

This authorizes the SDK to access your account resources.

---

## 2. Understanding Projects & Billing

Google Cloud organizes resources into **Projects**. A project is like a container for your API keys, compute instances, and quotas.
**Billing Accounts** are separate entities linked to projects to pay for usage.

### Listing Your Projects

To see all projects you have access to:

```bash
gcloud projects list
```

### Checking Billing Status

A project must be linked to a billing account to use most paid APIs (or generous free tiers like Gemini 3).
To check if a specific project has billing enabled:

```bash
gcloud beta billing projects describe <PROJECT_ID>
```

_Look for `billingEnabled: true` in the output._

### Listing Billing Accounts

To see available billing accounts you can link to:

```bash
gcloud beta billing accounts list
```

---

## 3. Demystifying `gcloud` Configuration

The `gcloud` CLI can differ between **who you are** (Auth) and **which settings you are using** (Configuration).

### Key Concepts

1.  **Auth List (`gcloud auth list`)**:

    - Shows all the Google Accounts (email addresses) you have logged in with.
    - Asterisk (\*) indicates the currently active account.

2.  **Configuration Configurations**:

    - A "Configuration" is a named profile storing your preferences (active project, region, active account).
    - You can have one config for "Work" and one for "Personal".

    **List Configurations:**

    ```bash
    gcloud config configurations list
    ```

    **Create a New Configuration:**

    ```bash
    gcloud config configurations create <NAME>
    ```

    **Activate a Configuration:**

    ```bash
    gcloud config configurations activate <NAME>
    ```

3.  **Setting the Active Project**:
    - Tells `gcloud` which project to execute commands against by default.
    ```bash
    gcloud config set project <PROJECT_ID>
    ```

---

## 4. Enabling the Gemini API

Before you can use the models, you must enable the "Generative Language" service for your project.

```bash
gcloud services enable generativelanguage.googleapis.com
```

---

## 5. Checking Gemini 3 Quotas

Access to newer models like **Gemini 3** is often gated by quotas before the models appear in public lists. You can verify your access by checking the API quotas directly.

### Command

Replace `<PROJECT_ID>` with your actual project ID.

```bash
gcloud alpha services quota list \
    --service=generativelanguage.googleapis.com \
    --consumer=projects/keen-jigsaw-484410-t4 \
    --format="table(metric, limit, usage)" | grep "gemini-3"
```

- **If you see output:** You have quota entries for Gemini 3 (e.g., `gemini-3-flash`, `gemini-3-pro`), meaning you have access.
- **If empty:** You may not have access to these models yet on this project.

---

## 6. Creating an API Key

Once confirmed, you need an API Key to make requests from your application.

### A. Enable API Keys Service

First, ensure the tool to create keys is enabled:

```bash
gcloud services enable apikeys.googleapis.com
```

### B. Creating the Key

You can choose either the Google Cloud Console (GUI) or the command line (CLI) to create your key.

#### Method 1: The Standard Way (Google Cloud Console)

Most API keys are stored in the Credentials section of your project.

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Select your project from the dropdown menu in the top navigation bar (if it's not already selected).
3.  Click the **Navigation Menu** (the three horizontal lines $\equiv$) in the top-left corner.
4.  Hover over **APIs & Services** and select **Credentials**.
5.  Look for the section titled **API Keys**.
6.  **If you see a key**: Click the **Show key** icon (eye symbol) to reveal it or the **Copy** icon to copy it.
7.  **If you don't see a key**: Click **+ CREATE CREDENTIALS** at the top of the page and select **API key**.

#### Method 2: The CLI Way

Create a new API key directly from your terminal with a display name:

```bash
gcloud alpha services api-keys create --display-name="Gemini 3 App Key"
```

**Output:**
Look for the `keyString` field in the JSON output. This is your `GEMINI_API_KEY`.

### C. (Optional) Restrict the Key

For security, it is best practice to restrict the key to only the Gemini API.

#### Using CLI

1.  Get the key's unique ID (UID) from the list:
    ```bash
    gcloud alpha services api-keys list
    ```
2.  Restrict it:
    ```bash
    gcloud alpha services api-keys update <KEY_UID> \
        --api-target=service=generativelanguage.googleapis.com
    ```

---

## Summary Cheat Sheet

```bash
# 1. Login
gcloud auth login

# 2. Select Project
gcloud config set project <YOUR_PROJECT_ID>

# 3. Enable API
gcloud services enable generativelanguage.googleapis.com

# 4. Check Gemini 3 Access
gcloud alpha services quota list \
    --service=generativelanguage.googleapis.com \
    --consumer=projects/$(gcloud config get-value project) \
    --format="json" | grep "gemini-3"

# 5. Create Key
gcloud services enable apikeys.googleapis.com
gcloud alpha services api-keys create --display-name="Gemini Key"
```
