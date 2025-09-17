// Typewriter effect
const typewriterElement = document.getElementById("typewriter");
const words = ["Prediction", "Analysis", "Intelligence"];
let wordIndex = 0;
let charIndex = 0;
let isDeleting = false;
let typingDelay = 100;
let deletingDelay = 50;
let pauseDelay = 1500;

function type() {
  const currentWord = words[wordIndex];

  if (isDeleting) {
    typewriterElement.textContent = currentWord.substring(0, charIndex - 1);
    charIndex--;
    typingDelay = deletingDelay;
  } else {
    typewriterElement.textContent = currentWord.substring(0, charIndex + 1);
    charIndex++;
    typingDelay = 100;
  }

  if (!isDeleting && charIndex === currentWord.length) {
    typingDelay = pauseDelay;
    isDeleting = true;
  } else if (isDeleting && charIndex === 0) {
    isDeleting = false;
    wordIndex = (wordIndex + 1) % words.length;
    typingDelay = 500;
  }

  setTimeout(type, typingDelay);
}

// Initialize typewriter effect
setTimeout(type, 1000);

// Sticky header
window.addEventListener("scroll", function () {
  const header = document.getElementById("header");
  if (window.scrollY > 50) {
    header.classList.add("shrink");
  } else {
    header.classList.remove("shrink");
  }
});

// Mobile navigation
const hamburger = document.getElementById("hamburger");
const navMenu = document.getElementById("nav-menu");

hamburger.addEventListener("click", function () {
  hamburger.classList.toggle("active");
  navMenu.classList.toggle("show");
});

// Form validation
document
  .getElementById("prediction-form")
  .addEventListener("submit", function (event) {
    const screenSizeInput = document.getElementById("screen_size");
    const screenSizeError = document.getElementById("screenSizeError");
    const weightInput = document.getElementById("weight");
    const weightError = document.getElementById("weightError");
    const hddInput = document.getElementById("HDD");
    const ssdInput = document.getElementById("SSD");
    const storageError = document.getElementById("storageError");

    let isValid = true;

    // Clear previous errors
    screenSizeError.textContent = "";
    weightError.textContent = "";
    storageError.textContent = "";

    // Validate Screen Size
    const screenSizeValue = parseFloat(screenSizeInput.value);
    if (
      isNaN(screenSizeValue) ||
      screenSizeValue < 10 ||
      screenSizeValue > 17
    ) {
      screenSizeError.textContent =
        "Screen size must be between 10 and 17 inches.";
      isValid = false;
    }

    // Validate Weight
    const weightValue = parseFloat(weightInput.value);
    if (isNaN(weightValue)) {
      weightError.textContent = "Please enter a valid weight.";
      isValid = false;
    } else if (weightValue < 1 || weightValue > 4) {
      weightError.textContent = "Weight must be between 1 kg and 4 kg.";
      isValid = false;
    }

    // Validate HDD or SSD Selection
    const hddValue = parseInt(hddInput.value);
    const ssdValue = parseInt(ssdInput.value);
    if (hddValue === 0 && ssdValue === 0) {
      storageError.textContent =
        "You must select a non-zero value for either HDD or SSD.";
      isValid = false;
    }

    if (!isValid) {
      event.preventDefault();
    }

    return isValid;
  });

// Save Recommendation
function saveRecommendation(button) {
  const laptopName = button.getAttribute("data-laptop-name");
  const specs = button.getAttribute("data-specs");
  const price = button.getAttribute("data-price");
  const similarity = button.getAttribute("data-similarity");

  // Disable button and show loading state
  button.disabled = true;
  button.innerHTML = '<span class="loading"></span>Saving...';

  fetch("/save_recommendation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      laptop_name: laptopName,
      specs: specs,
      price: price,
      similarity_score: parseFloat(similarity) || 0,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        button.textContent = "Saved";
        button.classList.add("saved");
        showFlashMessage("success", data.message);
      } else {
        button.disabled = false;
        button.textContent = "Save Recommendation";
        showFlashMessage("error", data.message);
      }
    })
    .catch((error) => {
      console.error("Error saving recommendation:", error);
      button.disabled = false;
      button.textContent = "Save Recommendation";
      showFlashMessage(
        "error",
        "An error occurred while saving the recommendation."
      );
    });
}

// Flash message display function
function showFlashMessage(category, message) {
  const flashContainer = document.querySelector(".flash-messages");
  const flashMessage = document.createElement("div");
  flashMessage.className = `flash-message flash-${category}`;
  flashMessage.innerHTML = `
    ${message}
    <span class="flash-close">&times;</span>
  `;
  flashContainer.appendChild(flashMessage);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    flashMessage.style.opacity = "0";
    setTimeout(() => flashMessage.remove(), 300);
  }, 5000);

  // Close button functionality
  flashMessage.querySelector(".flash-close").addEventListener("click", () => {
    flashMessage.style.opacity = "0";
    setTimeout(() => flashMessage.remove(), 300);
  });
}

// Initialize animations when elements come into view
const observerOptions = {
  root: null,
  rootMargin: "0px",
  threshold: 0.1,
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("animate");
    }
  });
}, observerOptions);

// Observe elements for animation
document
  .querySelectorAll(
    ".feature-card, .step, .prediction-result, .example-laptop, .recommended-laptop"
  )
  .forEach((el) => {
    observer.observe(el);
  });
