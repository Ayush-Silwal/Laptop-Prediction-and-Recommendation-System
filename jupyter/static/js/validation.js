
        document.querySelector('form').addEventListener('submit', function (event) {
            var screenSizeInput = document.getElementById('screen_size');
            var screenSizeError = document.getElementById('screenSizeError');
            var weightInput = document.getElementById('weight');
            var weightError = document.getElementById('weightError');

            var isValid = true;

            // Validate Screen Size
            if (screenSizeInput.value <= 0) {
                event.preventDefault(); // Prevent form submission
                screenSizeError.textContent = "Screen size cannot be zero. Recommended range is from 11 to 17 inches.";
                isValid = false;
            } else {
                screenSizeError.textContent = ""; // Clear the error message if valid
            }

            // Validate Weight
            var weightValue = parseFloat(weightInput.value);
            if (isNaN(weightValue) || weightValue <= 0) {
                event.preventDefault(); // Prevent form submission
                weightError.textContent = "Weight must be a positive number.";
                isValid = false;
            } else {
                weightError.textContent = ""; // Clear the error message if valid
            }

            return isValid;
        });

        window.onload = function() {
            document.querySelector('form').reset();
        };
 