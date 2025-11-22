import { CheckSquare as LucideCheckSquare, LucideProps } from 'lucide-react';

const TaskIcon = ({ className, ...props }: LucideProps) => {
  return <LucideCheckSquare className={className} {...props} />;
};

export default TaskIcon;