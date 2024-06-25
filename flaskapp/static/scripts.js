// Ensure DOM is fully loaded
document.addEventListener('DOMContentLoaded', (event) => {
    console.log('DOM fully loaded and parsed');

    // Handle form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach((form) => {
        form.addEventListener('submit', (event) => {
            event.preventDefault(); // Prevent default form submission
            const formData = new FormData(form);
            const action = form.getAttribute('action');
            const method = form.getAttribute('method');

            // Make an AJAX request to handle the form submission
            fetch(action, {
                method: method,
                body: formData,
            })
            .then(response => response.text())
            .then(data => {
                // Display the results in the container
                const container = document.querySelector('.container');
                container.innerHTML = data;
            })
            .catch(error => console.error('Error:', error));
        });
    });
});
