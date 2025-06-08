// Deep Space Aurora Theme JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Cursor glow effect
    const cursorGlow = document.getElementById('cursor-glow');
    let mouseTimeout;
    
    if (cursorGlow) {
        document.body.addEventListener('mouseenter', () => {
            cursorGlow.style.opacity = '1';
        });
        
        document.body.addEventListener('mouseleave', () => {
            cursorGlow.style.opacity = '0';
        });
        
        window.addEventListener('mousemove', (e) => {
            clearTimeout(mouseTimeout);
            cursorGlow.style.opacity = '1';
            
            gsap.to(cursorGlow, {
                duration: 1.5,
                x: e.clientX,
                y: e.clientY,
                ease: 'power3.out'
            });
            
            mouseTimeout = setTimeout(() => {
                cursorGlow.style.opacity = '0';
            }, 2000);
        });
    }

    // Particle background configuration - refined for Deep Space Aurora theme
    tsParticles.load("particles-js", {
        fullScreen: {
            enable: true,
            zIndex: -10
        },
        particles: {
            number: {
                value: 120,
                density: {
                    enable: true,
                    value_area: 800
                }
            },
            color: {
                value: "#ffffff"
            },
            shape: {
                type: "circle"
            },
            opacity: {
                value: 0.3,
                random: true,
                anim: {
                    enable: true,
                    speed: 0.5,
                    opacity_min: 0.1,
                    sync: false
                }
            },
            size: {
                value: 0.8,
                random: true,
                anim: {
                    enable: false
                }
            },
            line_linked: {
                enable: false
            },
            move: {
                enable: true,
                speed: 0.3,
                direction: "none",
                random: true,
                straight: false,
                out_mode: "out",
                bounce: false
            }
        },
        interactivity: {
            detect_on: "canvas",
            events: {
                onhover: {
                    enable: false
                },
                onclick: {
                    enable: false
                },
                resize: true
            }
        },
        retina_detect: true
    });
});

// Global variable to hold the plan data
let currentPlanData;

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offsetTop = target.offsetTop - 80;
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

// Tab functionality
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Consultation form submission
document.getElementById('consultation-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        company: document.getElementById('company').value,
        challenge: document.getElementById('challenge').value
    };
    
    console.log('Consultation form submitted:', formData);
    alert('Thank you for your interest! We\'ll be in touch within 24 hours to schedule your consultation.');
    this.reset();
});

// AI PLAN REVIEW RENDERING FUNCTIONS
function createTaskItem(task, phaseIndex, taskIndex) {
    const item = document.createElement('div');
    item.className = 'review-item relative glass-card p-6 rounded-lg transition-all duration-300 hover:transform hover:scale-102';
    item.dataset.phaseIndex = phaseIndex;
    item.dataset.taskIndex = taskIndex;

    const createInputGroup = (paramKey, value, type = 'number', step = '1') => {
        const paramId = `task-${phaseIndex}-${taskIndex}-${paramKey.replace('.', '-')}`;
        
        let labelText;
        const keyParts = paramKey.split('.');
        const mainKey = keyParts[0];
        const subKey = keyParts[1];

        const paramNameMap = {
            'optimistic': 'Optimistic', 'most_likely': 'Most Likely', 'pessimistic': 'Pessimistic',
            'mean': 'Mean', 'std_dev': 'Standard Deviation', 'min': 'Minimum', 'max': 'Maximum'
        };
        const unitMap = { 'duration_params': 'Duration (Days)', 'cost_params': 'Cost ($)' };

        if (paramNameMap[subKey] && unitMap[mainKey]) {
            labelText = `${paramNameMap[subKey]} ${unitMap[mainKey]}`;
        } else {
            labelText = (paramKey.charAt(0).toUpperCase() + paramKey.slice(1)).replace(/[_\.]/g, ' ');
        }

        return `
            <div class="form-group">
                <label for="${paramId}" class="form-label">${labelText}</label>
                <input type="${type}" id="${paramId}" value="${value}" step="${step}" data-key="${paramKey}" class="form-input">
            </div>
        `;
    };

    let paramsHtml = '';
    if (task.duration_params) {
        for (const [key, value] of Object.entries(task.duration_params)) {
            if (key !== 'type') {
                paramsHtml += createInputGroup(`duration_params.${key}`, value, 'number', '1');
            }
        }
    }
    if (task.cost_params) {
         for (const [key, value] of Object.entries(task.cost_params)) {
            if (key !== 'type') {
                paramsHtml += createInputGroup(`cost_params.${key}`, value, 'number', '0.01');
            }
        }
    }

    item.innerHTML = `
        <button class="delete-button absolute top-3 right-3 w-8 h-8 bg-red-500/80 hover:bg-red-500 rounded-full text-white text-sm font-bold transition-all duration-300 flex items-center justify-center" aria-label="Delete task">
            <i class="fas fa-times"></i>
        </button>
        <div class="review-item-header mb-4">
            <label for="task-name-${phaseIndex}-${taskIndex}" class="form-label">Task Name</label>
            <input type="text" id="task-name-${phaseIndex}-${taskIndex}" data-key="name" value="${task.name}" class="form-input text-lg font-semibold">
        </div>
        <div class="review-item-body grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            ${paramsHtml}
        </div>
    `;
    return item;
}

function createRiskItem(risk, riskIndex) {
    const item = document.createElement('div');
    item.className = 'review-item relative glass-card p-6 rounded-lg transition-all duration-300 hover:transform hover:scale-102';
    item.dataset.riskIndex = riskIndex;

    const createInputGroup = (paramKey, value, type = 'number', min = 0, max = undefined, step = '1') => {
        const paramId = `risk-${riskIndex}-${paramKey.replace(/\./g, '-')}`;
        const maxAttr = max !== undefined ? `max="${max}"` : '';
        
        let labelText;
        if (paramKey === 'probability') {
            labelText = 'Probability of Occurrence (%)';
        } else {
            const keyParts = paramKey.split('.');
            const mainKey = keyParts[0];
            const subKey = keyParts[1];

            const paramNameMap = {
                'optimistic': 'Optimistic', 'most_likely': 'Most Likely', 'pessimistic': 'Pessimistic',
                'mean': 'Mean', 'std_dev': 'Standard Deviation', 'min': 'Minimum', 'max': 'Maximum'
            };
            const unitMap = { 'impact_days': 'Impact (Days)', 'impact_cost': 'Impact (Cost $)' };

            if (paramNameMap[subKey] && unitMap[mainKey]) {
                labelText = `${paramNameMap[subKey]} ${unitMap[mainKey]}`;
            } else {
                labelText = (paramKey.charAt(0).toUpperCase() + paramKey.slice(1)).replace(/[_\.]/g, ' ');
            }
        }

        return `
            <div class="form-group">
                <label for="${paramId}" class="form-label">${labelText}</label>
                <input type="${type}" id="${paramId}" value="${value}" min="${min}" ${maxAttr} step="${step}" data-key="${paramKey}" class="form-input">
            </div>
        `;
    };
    
    let paramsHtml = createInputGroup('probability', risk.probability, 'number', 0, 100);

    if (risk.impact_days) {
        for (const [key, value] of Object.entries(risk.impact_days)) {
            if (key !== 'type') {
                paramsHtml += createInputGroup(`impact_days.${key}`, value);
            }
        }
    }

    if (risk.impact_cost) {
        for (const [key, value] of Object.entries(risk.impact_cost)) {
            if (key !== 'type') {
                paramsHtml += createInputGroup(`impact_cost.${key}`, value, 'number', 0, undefined, '0.01');
            }
        }
    }

    item.innerHTML = `
        <button class="delete-button absolute top-3 right-3 w-8 h-8 bg-red-500/80 hover:bg-red-500 rounded-full text-white text-sm font-bold transition-all duration-300 flex items-center justify-center" aria-label="Delete risk">
            <i class="fas fa-times"></i>
        </button>
        <div class="review-item-header mb-4">
            <label for="risk-name-${riskIndex}" class="form-label">Risk Name</label>
            <input type="text" id="risk-name-${riskIndex}" data-key="name" value="${risk.name}" class="form-input text-lg font-semibold">
        </div>
        <div class="review-item-body grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            ${paramsHtml}
        </div>
    `;
    return item;
}

function renderPlanForReview(planData) {
    const tasksListContainer = document.getElementById('tasks-list');
    const risksListContainer = document.getElementById('risks-list');

    tasksListContainer.innerHTML = '';
    risksListContainer.innerHTML = '';

    if (planData && planData.phases && Array.isArray(planData.phases)) {
        planData.phases.forEach((phase, phaseIndex) => {
            const phaseContainer = document.createElement('div');
            phaseContainer.className = 'phase-container mb-8';

            const phaseHeading = document.createElement('h3');
            phaseHeading.className = 'text-xl font-semibold text-light-text mb-4';
            phaseHeading.textContent = phase.phaseName;
            phaseContainer.appendChild(phaseHeading);

            if (phase.tasks && Array.isArray(phase.tasks)) {
                phase.tasks.forEach((task, taskIndex) => {
                    const taskElement = createTaskItem(task, phaseIndex, taskIndex);
                    phaseContainer.appendChild(taskElement);
                });
            }
            
            const addTaskButton = document.createElement('button');
            addTaskButton.className = 'add-item-button add-task-button mt-4 w-full';
            addTaskButton.textContent = 'Add Task to This Phase';
            addTaskButton.dataset.phaseIndex = phaseIndex;
            phaseContainer.appendChild(addTaskButton);

            tasksListContainer.appendChild(phaseContainer);
        });
    }

    if (planData && planData.risks && Array.isArray(planData.risks)) {
        planData.risks.forEach((risk, riskIndex) => {
            const riskElement = createRiskItem(risk, riskIndex);
            risksListContainer.appendChild(riskElement);
        });
    }
}

// INTERACTIVE PLAN EVENT LISTENERS
document.addEventListener('DOMContentLoaded', () => {
    const reviewSection = document.getElementById('review-plan-section');
    if (!reviewSection) return;

    reviewSection.addEventListener('click', function(e) {
        const target = e.target;
        
        if (target.matches('.delete-button') || target.closest('.delete-button')) {
            const deleteBtn = target.closest('.delete-button');
            const reviewItem = deleteBtn.closest('.review-item');
            if (!reviewItem) return;

            if (reviewItem.dataset.taskIndex !== undefined) {
                const phaseIndex = parseInt(reviewItem.dataset.phaseIndex, 10);
                const taskIndex = parseInt(reviewItem.dataset.taskIndex, 10);
                currentPlanData.phases[phaseIndex].tasks.splice(taskIndex, 1);
            } else if (reviewItem.dataset.riskIndex !== undefined) {
                const riskIndex = parseInt(reviewItem.dataset.riskIndex, 10);
                currentPlanData.risks.splice(riskIndex, 1);
            }
            renderPlanForReview(currentPlanData);
        }

        if (target.matches('.add-task-button')) {
            const phaseIndex = parseInt(target.dataset.phaseIndex, 10);
            if (currentPlanData && currentPlanData.phases[phaseIndex]) {
                const newTask = {
                    id: `NewTask-${Date.now()}`,
                    name: "New Task",
                    duration_params: { type: "PERT", optimistic: 5, most_likely: 10, pessimistic: 15 },
                    cost_params: { type: "PERT", optimistic: 1000, most_likely: 2000, pessimistic: 4000 }
                };
                currentPlanData.phases[phaseIndex].tasks.push(newTask);
                renderPlanForReview(currentPlanData);
            }
        }

        if (target.matches('#add-risk-button')) {
            if (currentPlanData && currentPlanData.risks) {
                 const newRisk = {
                    id: `NewRisk-${Date.now()}`,
                    name: "New Risk",
                    probability: 20,
                    impact_days: { type: "PERT", optimistic: 2, most_likely: 5, pessimistic: 10 },
                    impact_cost: { type: "PERT", optimistic: 5000, most_likely: 10000, pessimistic: 20000 }
                };
                currentPlanData.risks.push(newRisk);
                renderPlanForReview(currentPlanData);
            }
        }
    });

    reviewSection.addEventListener('input', function(e) {
        if (e.target.tagName === 'INPUT' && e.target.dataset.key) {
            const reviewItem = e.target.closest('.review-item');
            if (!reviewItem) return;
            
            const keys = e.target.dataset.key.split('.');
            const value = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;

            if (reviewItem.dataset.taskIndex !== undefined) {
                const phaseIndex = parseInt(reviewItem.dataset.phaseIndex, 10);
                const taskIndex = parseInt(reviewItem.dataset.taskIndex, 10);
                const taskObject = currentPlanData.phases[phaseIndex].tasks[taskIndex];
                if (keys.length > 1) {
                    taskObject[keys[0]][keys[1]] = value;
                } else {
                    taskObject[keys[0]] = value;
                }
            } else if (reviewItem.dataset.riskIndex !== undefined) {
                const riskIndex = parseInt(reviewItem.dataset.riskIndex, 10);
                const riskObject = currentPlanData.risks[riskIndex];
                if (keys.length > 1) {
                    riskObject[keys[0]][keys[1]] = value;
                } else {
                    riskObject[keys[0]] = value;
                }
            }
            console.log('Plan data updated:', currentPlanData);
        }
    });
});

// ENHANCED CLIENT-SIDE VALIDATION
const validationState = {
    projectName: false,
    baseCost: false,
    baseDuration: false,
    projectType: false
};

const simulationForm = document.getElementById('simulation-form');
const projectNameInput = document.getElementById('project-name');
const baseCostInput = document.getElementById('base-cost');
const baseDurationInput = document.getElementById('base-duration');
const projectTypeSelect = document.getElementById('project-type');
const submitButton = document.getElementById('run-simulation-button');

const projectNameError = document.getElementById('project-name-error');
const baseCostError = document.getElementById('base-cost-error');
const baseDurationError = document.getElementById('base-duration-error');
const projectTypeError = document.getElementById('project-type-error');

function validateProjectName() {
    const value = projectNameInput.value;
    const trimmedValue = value.trim();
    
    if (!trimmedValue) {
        showError(projectNameInput, projectNameError, 'Project name is required and cannot be empty.');
        validationState.projectName = false;
        return false;
    }
    
    if (value.length > 100) {
        showError(projectNameInput, projectNameError, 'Project name must not exceed 100 characters.');
        validationState.projectName = false;
        return false;
    }
    
    clearError(projectNameInput, projectNameError);
    validationState.projectName = true;
    return true;
}

function validateBaseCost() {
    const value = baseCostInput.value;
    
    if (value === '') {
        showError(baseCostInput, baseCostError, 'Base cost is required.');
        validationState.baseCost = false;
        return false;
    }
    
    const numValue = parseFloat(value);
    
    if (isNaN(numValue)) {
        showError(baseCostInput, baseCostError, 'Please enter a valid number.');
        validationState.baseCost = false;
        return false;
    }
    
    if (numValue < 0) {
        showError(baseCostInput, baseCostError, 'Base cost must be a non-negative number.');
        validationState.baseCost = false;
        return false;
    }
    
    clearError(baseCostInput, baseCostError);
    validationState.baseCost = true;
    return true;
}

function validateBaseDuration() {
    const value = baseDurationInput.value;
    
    if (value === '') {
        showError(baseDurationInput, baseDurationError, 'Base duration is required.');
        validationState.baseDuration = false;
        return false;
    }
    
    const numValue = parseFloat(value);
    
    if (isNaN(numValue)) {
        showError(baseDurationInput, baseDurationError, 'Please enter a valid number.');
        validationState.baseDuration = false;
        return false;
    }
    
    if (!Number.isInteger(numValue)) {
        showError(baseDurationInput, baseDurationError, 'Base duration must be a whole number (integer).');
        validationState.baseDuration = false;
        return false;
    }
    
    if (numValue < 1) {
        showError(baseDurationInput, baseDurationError, 'Base duration must be at least 1 day.');
        validationState.baseDuration = false;
        return false;
    }
    
    clearError(baseDurationInput, baseDurationError);
    validationState.baseDuration = true;
    return true;
}

function validateProjectType() {
    const value = projectTypeSelect.value;
    
    if (!value) {
        showError(projectTypeSelect, projectTypeError, 'Please select a project type.');
        validationState.projectType = false;
        return false;
    }
    
    clearError(projectTypeSelect, projectTypeError);
    validationState.projectType = true;
    return true;
}

function showError(inputElement, errorElement, message) {
    inputElement.classList.add('error');
    errorElement.textContent = message;
    errorElement.classList.add('show');
    
    inputElement.setAttribute('aria-invalid', 'true');
    inputElement.setAttribute('aria-describedby', errorElement.id);
}

function clearError(inputElement, errorElement) {
    inputElement.classList.remove('error');
    errorElement.textContent = '';
    errorElement.classList.remove('show');
    
    inputElement.setAttribute('aria-invalid', 'false');
    inputElement.removeAttribute('aria-describedby');
}

function updateSubmitButtonState() {
    const allValid = Object.values(validationState).every(isValid => isValid);
    submitButton.disabled = !allValid;
}

function validateAllFields() {
    validateProjectName();
    validateBaseCost();
    validateBaseDuration();
    validateProjectType();
    updateSubmitButtonState();
}

// Add event listeners for real-time validation
projectNameInput.addEventListener('input', () => {
    validateProjectName();
    updateSubmitButtonState();
});

projectNameInput.addEventListener('blur', () => {
    validateProjectName();
    updateSubmitButtonState();
});

baseCostInput.addEventListener('input', () => {
    validateBaseCost();
    updateSubmitButtonState();
});

baseCostInput.addEventListener('blur', () => {
    validateBaseCost();
    updateSubmitButtonState();
});

baseDurationInput.addEventListener('input', () => {
    validateBaseDuration();
    updateSubmitButtonState();
});

baseDurationInput.addEventListener('blur', () => {
    validateBaseDuration();
    updateSubmitButtonState();
});

projectTypeSelect.addEventListener('change', () => {
    validateProjectType();
    updateSubmitButtonState();
});

projectTypeSelect.addEventListener('blur', () => {
    validateProjectType();
    updateSubmitButtonState();
});

// Perform initial validation on page load
window.addEventListener('DOMContentLoaded', () => {
    validateAllFields();
});

// Simulation form submission
simulationForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    validateAllFields();
    
    if (!Object.values(validationState).every(isValid => isValid)) {
        if (!validationState.projectName) projectNameInput.focus();
        else if (!validationState.baseCost) baseCostInput.focus();
        else if (!validationState.baseDuration) baseDurationInput.focus();
        else if (!validationState.projectType) projectTypeSelect.focus();
        
        return;
    }
    
    const statusMessageDiv = document.getElementById('status-message');
    const submitButton = document.getElementById('run-simulation-button');
    
    submitButton.disabled = true;
    statusMessageDiv.textContent = 'Generating AI project plan, please wait...';
    statusMessageDiv.className = 'status-message pending';
    
    const projectName = projectNameInput.value.trim();
    const baseCost = parseFloat(baseCostInput.value);
    const baseDuration = parseInt(baseDurationInput.value);
    const projectType = projectTypeSelect.value;
    
    const simulationData = {
        projectName: projectName,
        baseCost: baseCost,
        baseDuration: baseDuration,
        projectType: projectType
    };
    
    fetch('http://localhost:5000/generate-plan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(simulationData)
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            return response.text().then(text => {
                throw new Error(`Server responded with status ${response.status}: ${text}`);
            });
        }
    })
    .then(data => {
        currentPlanData = data;
        console.log('Plan generated:', currentPlanData);

        const simulationSection = document.getElementById('simulation');
        const reviewSection = document.getElementById('review-plan-section');

        simulationSection.style.display = 'none';
        reviewSection.style.display = 'block';

        renderPlanForReview(currentPlanData);

        reviewSection.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
        let errorMessage;
        if (error.message.includes('fetch')) {
            errorMessage = `Network error or server unavailable. Please check your connection and ensure the local server is running. Error: ${error.message}`;
        } else {
            errorMessage = `Error: Plan generation failed. ${error.message}`;
        }
        statusMessageDiv.textContent = errorMessage;
        statusMessageDiv.className = 'status-message error';
    })
    .finally(() => {
        submitButton.disabled = false;
        validateAllFields();
    });
});

// Event listener for the final simulation button
const finalSimButton = document.getElementById('run-simulation-final-button');
const reviewStatusMessage = document.getElementById('review-status-message');

finalSimButton.addEventListener('click', function() {
    finalSimButton.disabled = true;
    reviewStatusMessage.textContent = 'Running final simulation with your plan...';
    reviewStatusMessage.className = 'status-message pending';
    reviewStatusMessage.style.display = 'flex';

    fetch('http://localhost:5000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentPlanData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { 
                const errorMessage = err.message || response.statusText;
                throw new Error(errorMessage);
            });
        }
        return response.json();
    })
    .then(data => {
        reviewStatusMessage.textContent = `Simulation Complete! Report '${data.report_path}' generated.`;
        reviewStatusMessage.className = 'status-message success';
        
        const reviewGrid = document.querySelector('.review-grid');
        if(reviewGrid) {
            reviewGrid.style.display = 'none';
        }
    })
    .catch(error => {
        reviewStatusMessage.textContent = `Error: ${error.message}`;
        reviewStatusMessage.className = 'status-message error';
        finalSimButton.disabled = false;
    });
});

// Navbar scroll effect
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(22, 27, 41, 0.98)';
        navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.3)';
    } else {
        navbar.style.background = 'rgba(22, 27, 41, 0.7)';
        navbar.style.boxShadow = 'none';
    }
    
    lastScroll = currentScroll;
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections
document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
});

// Mobile menu toggle
document.querySelector('.mobile-menu').addEventListener('click', function() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('active');
});

// Dynamic node animations
function animateNodes() {
    const nodes = document.querySelectorAll('.node');
    const connections = document.querySelectorAll('.connection');
    
    nodes.forEach((node, index) => {
        const delay = index * 0.2;
        node.style.animationDelay = `${delay}s`;
    });
    
    connections.forEach((connection, index) => {
        const delay = index * 0.3;
        connection.style.animationDelay = `${delay}s`;
    });
}

animateNodes();

// Add hover effects to pain points
document.querySelectorAll('.pain-point').forEach(point => {
    point.addEventListener('mouseenter', function() {
        this.querySelector('.pain-point-icon i').style.transform = 'scale(1.1)';
    });
    
    point.addEventListener('mouseleave', function() {
        this.querySelector('.pain-point-icon i').style.transform = 'scale(1)';
    });
});

// Add hover effects to advantages
document.querySelectorAll('.advantage').forEach(advantage => {
    advantage.addEventListener('mouseenter', function() {
        this.querySelector('.advantage-icon i').style.transform = 'rotate(10deg) scale(1.1)';
    });
    
    advantage.addEventListener('mouseleave', function() {
        this.querySelector('.advantage-icon i').style.transform = 'rotate(0) scale(1)';
    });
});

// Form field animations
document.querySelectorAll('.form-group input, .form-group textarea').forEach(field => {
    field.addEventListener('focus', function() {
        const label = this.parentElement.querySelector('label');
        if (label) label.style.color = '#2dd4bf';
    });
    
    field.addEventListener('blur', function() {
        const label = this.parentElement.querySelector('label');
        if (label) label.style.color = '#e0e6f0';
    });
});